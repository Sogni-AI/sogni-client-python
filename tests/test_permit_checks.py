import copy
import json
import time
from unittest.mock import patch

import httpx
import pytest
from eth_account import Account

from sogni_client.account import AccountApi
from sogni_client.errors import ApiError
from sogni_client.transport import ApiClient

WALLET = Account.from_key(bytes.fromhex("01" * 32))
CHAINS = {False: {"base": 8453, "etherlink": 42793}, True: {"base": 84532, "etherlink": 11155111}}
TOKENS = {
    "base": "0xE014d2A4da6E450f21b5050120D291e63c8940FD",
    "etherlink": "0x2767Fc1048cd0ae4fd952eeF09853301c9DC7173",
}
DOMAIN_FIELDS = [
    {"name": n, "type": t}
    for n, t in [
        ("name", "string"),
        ("version", "string"),
        ("chainId", "uint256"),
        ("verifyingContract", "address"),
    ]
]


def permit_payload(kind, testnet=False, provider="base"):
    domain = {
        "name": "Sogni-testnet" if testnet else "Sogni AI",
        "version": "1",
        "chainId": str(CHAINS[testnet][provider]),
        "verifyingContract": "0xCcCCccccCCCCcCCCCCCcCcCccCcCCCcCcccccccC",
    }
    if kind == "withdraw":
        fields = [
            ("walletAddress", "address"),
            ("amount", "string"),
            ("provider", "string"),
            ("nonce", "string"),
        ]
        message = {
            "walletAddress": WALLET.address,
            "amount": "1250000000000000000",
            "provider": provider,
            "nonce": "example",
        }
        name = "WithdrawTokenV2"
    else:
        domain["verifyingContract"] = TOKENS[provider] if not testnet else "0x" + "12" * 20
        fields = [
            ("owner", "address"),
            ("spender", "address"),
            ("value", "uint256"),
            ("nonce", "uint256"),
            ("deadline", "uint256"),
        ]
        message = {
            "owner": WALLET.address,
            "spender": "0x" + "34" * 20,
            "value": str(2**256 - 1),
            "nonce": 0,
            "deadline": int(time.time()) + 7 * 86400,
        }
        name = "Permit"
    return {
        "domain": domain,
        "types": {name: [{"name": n, "type": t} for n, t in fields]},
        "message": message,
    }


async def invoke(
    kind,
    permit,
    *,
    testnet=False,
    provider="base",
    rejected=False,
    http_class=httpx.AsyncClient,
    proof=None,
):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(
            200, json={"data": permit if request.url.path.endswith("/permit") else {}}
        )

    async with http_class(transport=httpx.MockTransport(handle)) as http:
        client = ApiClient(
            base_url="https://api.example",
            socket_url="wss://socket.example",
            app_id="test",
            network="fast",
            auth_type="token",
            disable_socket=True,
            http_client=http,
        )
        account = AccountApi(client, testnet=testnet)
        account.current_account._update({"username": "test", "wallet_address": WALLET.address})
        with (
            patch.object(account, "get_wallet", return_value=WALLET),
            patch(
                "sogni_client.account._sign_typed_data",
                wraps=__import__(
                    "sogni_client.account", fromlist=["_sign_typed_data"]
                )._sign_typed_data,
            ) as sign,
        ):
            args = ("unused", "1.25" if kind == "withdraw" else "account", provider)
            operation = account.withdraw if kind == "withdraw" else account.approve_token_usage
            if rejected:
                with pytest.raises(ApiError) as caught:
                    await operation(*args)
                assert caught.value.status == 400
                assert caught.value.payload == {
                    "status": "error",
                    "message": "Unexpected signing request; nothing was signed",
                    "errorCode": 0,
                }
                assert caught.value.error_code == 0
                sign.assert_not_called()
                assert all(call.url.path.endswith("/permit") for call in calls)
            else:
                await operation(*args, **({"proof": proof} if proof is not None else {}))
                sign.assert_called_once()
                assert len(calls) == 2
                body = json.loads(calls[-1].content)
                assert len(body.get("signature", body.get("approveSignature"))) == 132
        await client.aclose()
    return calls


@pytest.mark.parametrize(
    "proof", [None, {"code": "012345"}, {"challengeId": "c1", "assertion": {"id": "key"}}]
)
async def test_withdraw_sends_proof_only_in_signed_submission(proof):
    permit, submit = await invoke("withdraw", permit_payload("withdraw"), proof=proof)
    assert "proof" not in json.loads(permit.content)
    assert submit.url.path == "/v2/account/token/withdraw"
    body = json.loads(submit.content)
    if proof is None:
        assert "proof" not in body
    else:
        assert body["proof"] == proof


@pytest.mark.parametrize("kind", ["withdraw", "approve"])
async def test_oversized_chain_uses_plain_error(kind):
    permit = permit_payload(kind)
    permit["domain"]["chainId"] = "9" * 5000
    calls = await invoke(kind, permit, rejected=True)
    assert len(calls) == 1


@pytest.mark.parametrize("kind", ["withdraw", "approve"])
@pytest.mark.parametrize("testnet", [False, True])
@pytest.mark.parametrize("provider", ["Base", "BASE", "Etherlink", "other", ""])
async def test_other_providers_expect_base_like_the_api(kind, testnet, provider):
    # sogni-api uses the Etherlink domain only for exactly "etherlink"; any other value is Base.
    permit = permit_payload(kind, testnet, "base")
    if kind == "withdraw":
        permit["message"]["provider"] = provider
    calls = await invoke(kind, permit, testnet=testnet, provider=provider)
    assert json.loads(calls[0].content)["provider"] == provider
    await invoke(
        kind,
        permit_payload(kind, testnet, "etherlink"),
        testnet=testnet,
        provider=provider,
        rejected=True,
    )


@pytest.mark.parametrize("kind", ["withdraw", "approve"])
@pytest.mark.parametrize("testnet", [False, True])
@pytest.mark.parametrize("provider", ["base", "etherlink"])
@pytest.mark.parametrize("explicit_domain", [False, True])
async def test_expected_permit_signs(kind, testnet, provider, explicit_domain):
    permit = permit_payload(kind, testnet, provider)
    permit["domain"]["verifyingContract"] = permit["domain"]["verifyingContract"].lower()
    if explicit_domain:
        permit["types"]["EIP712Domain"] = copy.deepcopy(DOMAIN_FIELDS)
        permit["domain"]["chainId"] = int(permit["domain"]["chainId"])
    await invoke(kind, permit, testnet=testnet, provider=provider)


@pytest.mark.parametrize(
    "field,value",
    [
        ("domain.chainId", 1),
        ("domain.chainId", 8453.5),
        ("domain.verifyingContract", "0x" + "12" * 20),
        ("domain.name", "Other"),
        ("domain.version", "2"),
        ("domain.salt", "0x" + "00" * 32),
        ("types.Extra", [{"name": "value", "type": "string"}]),
        ("types.EIP712Domain", []),
        ("types.WithdrawTokenV2", []),
        ("message.amount", "1"),
        ("message.walletAddress", "0x" + "12" * 20),
        ("message.provider", "etherlink"),
        ("message.nonce", 123),
    ],
)
async def test_withdraw_checks_returned_field(field, value):
    permit = permit_payload("withdraw")
    section, key = field.split(".")
    permit[section][key] = value
    await invoke("withdraw", permit, rejected=True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("domain.chainId", 1),
        ("domain.verifyingContract", "0x" + "12" * 20),
        ("types.Extra", [{"name": "value", "type": "string"}]),
        ("types.Permit", []),
        ("types.EIP712Domain", []),
        ("message.owner", "0x" + "12" * 20),
        ("message.deadline", int(time.time()) + 31 * 86400),
        ("message.deadline", str(int(time.time()) + 31 * 86400)),
        ("message.deadline", "9999999999"),
        ("message.deadline", -1),
        ("message.deadline", "-1"),
        ("message.deadline", "12.5"),
        ("message.deadline", 12.0),
        ("message.deadline", None),
        ("message.deadline", True),
    ],
)
async def test_approve_checks_returned_field(field, value):
    permit = permit_payload("approve")
    section, key = field.split(".")
    permit[section][key] = value
    await invoke("approve", permit, rejected=True)


@pytest.mark.parametrize(
    "deadline",
    [
        0,
        int(time.time()) - 86400,
        int(time.time()) + 7 * 86400,
        int(time.time()) + 29 * 86400,
        str(int(time.time()) + 7 * 86400),
    ],
)
async def test_approve_deadline_allows_clock_skew(deadline):
    # sogni-api issues now + 7 days; a past or later deadline from a skewed clock still signs.
    permit = permit_payload("approve")
    permit["message"]["deadline"] = deadline
    calls = await invoke("approve", permit)
    assert json.loads(calls[-1].content)["deadline"] == deadline


@pytest.mark.parametrize("kind", ["withdraw", "approve"])
@pytest.mark.parametrize("section", ["domain", "types", "message"])
@pytest.mark.parametrize("value", [None, [], "other"])
async def test_permit_requires_objects(kind, section, value):
    permit = permit_payload(kind)
    permit[section] = value
    await invoke(kind, permit, rejected=True)


@pytest.mark.parametrize("kind", ["withdraw", "approve"])
async def test_permit_checks_apply_with_an_integrator_client(kind):
    class OwnRequest(httpx.AsyncClient):
        async def request(self, method, url, **kwargs):
            return await super().request(method, url, **kwargs)

    await invoke(kind, permit_payload(kind), http_class=OwnRequest)
    permit = permit_payload(kind)
    permit["domain"]["chainId"] = "1"
    await invoke(kind, permit, http_class=OwnRequest, rejected=True)
