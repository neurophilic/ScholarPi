// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

contract ScholarPi_PiQ_Token {
    string public name = "Pi Quotient";
    string public symbol = "piQ";
    uint8 public decimals = 18;
    
    address public admin;
    uint256 public totalSupply;

    mapping(address => uint256) public balanceOf;
    
    // THE REPLAY ATTACK DEFENSE: Permanently tracks assessed paper hashes
    mapping(string => bool) public hasBeenAssessed;

    // Events for the DeSci Index
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
     * @dev Mints piQ to the researcher. Reverts instantly if the evalHash was already assessed.
     */
    function verifyProofAndMint(
        address researcher, 
        uint256 amount, 
        string memory evalHash, 
        bytes memory zkProof
    ) public onlyAdmin {
        // 1. Double-Minting / Replay Attack Prevention
        require(!hasBeenAssessed[evalHash], "Fraud Detected: This manuscript hash has already claimed piQ.");
        
        // 2. Mock ZK-Proof Verification (In production, integrate Halo2/SnarkJS verifier here)
        require(zkProof.length > 0, "Invalid Zero-Knowledge Proof payload.");

        // 3. State Updates
        hasBeenAssessed[evalHash] = true;
        
        uint256 mintedAmount = amount * (10 ** uint256(decimals));
        balanceOf[researcher] += mintedAmount;
        totalSupply += mintedAmount;

        emit Mint(researcher, mintedAmount, evalHash);
    }

    /**
     * @dev Continuous Legitimacy Auditing: Slashing function triggered by human peer consensus.
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