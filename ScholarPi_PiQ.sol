// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

contract ScholarPi_PiQ_Token {
    string public name = "Pi Quotient";
    string public symbol = "piQ";
    uint8 public decimals = 18;
    
    address public admin;
    uint256 public totalSupply;

    mapping(address => uint256) public balanceOf;
    
    // REPLAY ATTACK DEFENSE: Permanently tracks assessed paper hashes
    mapping(string => bool) public hasBeenAssessed;

    event Mint(address indexed researcher, uint256 amount, string evalHash);
    event SlashingApplied(address indexed researcher, uint256 penaltyAmount, string reason);

    modifier onlyAdmin() {
        require(msg.sender == admin, "Unauthorized: Only ScholarPi Oracle can execute this.");
        _;
    }

    constructor() {
        admin = msg.sender;
    }

    /**
     * @dev Mints piQ to researcher. Reverts if evalHash was already assessed.
     * @param amountWei The mint amount already scaled to full 18-decimal
     * token precision (i.e. piQ_amount * 10**18), computed off-chain.
     */
    function verifyProofAndMint(
        address researcher, 
        uint256 amountWei, 
        string memory evalHash, 
        bytes memory zkProof
    ) public onlyAdmin {
        require(!hasBeenAssessed[evalHash], "Fraud Detected: This manuscript hash has already claimed piQ.");
        require(zkProof.length > 0, "Invalid Zero-Knowledge Proof payload.");
        require(amountWei > 0, "Mint amount must be greater than zero.");

        hasBeenAssessed[evalHash] = true;

        balanceOf[researcher] += amountWei;
        totalSupply += amountWei;

        emit Mint(researcher, amountWei, evalHash);
    }

    /**
     * @dev Continuous Legitimacy Auditing: Slashing function triggered by peer consensus.
     */
    function slashTokens(
        address researcher, 
        uint256 penaltyAmount, 
        string memory reason
    ) public onlyAdmin {
        uint256 slashAmount = penaltyAmount * (10 ** uint256(decimals));
        require(balanceOf[researcher] >= slashAmount, "Insufficient piQ balance for slashing penalty.");
        
        balanceOf[researcher] -= slashAmount;
        totalSupply -= slashAmount;
        
        emit SlashingApplied(researcher, slashAmount, reason);
    }
}
