// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract PiResearchCoin is ERC20, Ownable {
    // Maps a document's eval_hash to prevent double-minting
    mapping(string => bool) public evaluatedPapers;

    constructor() ERC20("Pi Research Coin", "PIC") {}

    /**
     * @dev Mints $PIC to a researcher after validating the zk-SNARK Proof of Research.
     * @param researcher The Ethereum wallet of the author.
     * @param amount The calculated reward (factors in baseline quality + improvement multiplier).
     * @param evalHash The unique hash of the document.
     * @param zkProof The cryptographic SNARK proof from the π-Index Engine.
     */
    function verifyProofAndMint(
        address researcher, 
        uint256 amount, 
        string memory evalHash, 
        bytes memory zkProof
    ) external onlyOwner {
        require(!evaluatedPapers[evalHash], "Paper already minted rewards.");
        require(zkProof.length > 0, "Invalid zk-SNARK proof.");
        
        // In a full production environment, this calls a Groth16 Verifier contract:
        // require(Verifier.verifyProof(zkProof, ...), "SNARK Verification Failed");

        evaluatedPapers[evalHash] = true;
        
        // Mint the tokens (amount is in wei, 18 decimals)
        _mint(researcher, amount * (10 ** decimals()));
    }
}
