"""Expected account permit domains and typed fields."""

# Chain IDs and withdrawal domains: sogni-api ecosystem.config.js and withdraw/permit.
# Mainnet token contracts: docs.sogni.ai, SOGNI token overview.
PERMIT_NETWORKS = {
    "mainnet": {
        "base": {"chainId": 8453, "token": "0xE014d2A4da6E450f21b5050120D291e63c8940FD"},
        "etherlink": {"chainId": 42793, "token": "0x2767Fc1048cd0ae4fd952eeF09853301c9DC7173"},
    },
    "testnet": {
        "base": {"chainId": 84532, "token": None},
        "etherlink": {"chainId": 11155111, "token": None},
    },
}
WITHDRAW_NAMES = {"mainnet": "Sogni AI", "testnet": "Sogni-testnet"}
WITHDRAW_VERSION = "1"
WITHDRAW_CONTRACT = "0xCcCCccccCCCCcCCCCCCcCcCccCcCCCcCcccccccC"

# EIP-712 standard domain fields, in canonical order.
DOMAIN_FIELDS = [
    {"name": "name", "type": "string"},
    {"name": "version", "type": "string"},
    {"name": "chainId", "type": "uint256"},
    {"name": "verifyingContract", "type": "address"},
]
# sogni-api /v1/account/token/withdraw/permit.
WITHDRAW_TYPES = {
    "WithdrawTokenV2": [
        {"name": "walletAddress", "type": "address"},
        {"name": "amount", "type": "string"},
        {"name": "provider", "type": "string"},
        {"name": "nonce", "type": "string"},
    ]
}
# sogni-api /v1/contract/token/approve/permit issues deadlines 7 days ahead; 30 days
# leaves room for a client clock that runs behind.
APPROVE_DEADLINE_WINDOW = 30 * 86400
# ERC-2612, returned by sogni-api /v1/contract/token/approve/permit.
APPROVE_TYPES = {
    "Permit": [
        {"name": "owner", "type": "address"},
        {"name": "spender", "type": "address"},
        {"name": "value", "type": "uint256"},
        {"name": "nonce", "type": "uint256"},
        {"name": "deadline", "type": "uint256"},
    ]
}
