//
//  GGUFModel.swift
//  MobileAIBench
//
//  Created by chaparro2001 on 9/11/25.
//

enum GGUFModel: String, CaseIterable {
    
    static func getModelByIndex(index: Int) -> GGUFModel {
        let allModels = GGUFModel.allCases
        guard index >= 0 && index < allModels.count else {
            return GGUFModel.allCases[0]
        }
        return allModels[index]
    }
    
    static var ggufModels: [GGUFModel] {
        return [
            .gemma_3_4b_it_q3_k_m,
            .gemma_3_4b_it_q3_k_s,
            .gemma_3_4b_it_q4_k_m,
            .gemma_3_4b_it_q4_k_s,
            .gemma_3_4b_it_q5_k_s,
            .gemma_3_4b_it_q5_k_m,
            .qwen3_4b_instruct_2507_q3_k_m,
            .qwen3_4b_instruct_2507_q3_k_s,
            .qwen3_4b_instruct_2507_q4_k_m,
            .qwen3_4b_instruct_2507_q4_k_s,
            .qwen3_4b_instruct_2507_q5_k_s,
            .qwen3_4b_instruct_2507_q5_k_m,
            .phi_2_Q4_K_M_1,
            .gemma_2b_it_Q4_K_M
        ]
    }
    case gemma_3_4b_it_q3_k_m = "gemma-3-4b-it-q3_k_m"
    case gemma_3_4b_it_q3_k_s = "gemma-3-4b-it-q3_k_s"
    case gemma_3_4b_it_q4_k_m = "gemma-3-4b-it-q4_k_m"
    case gemma_3_4b_it_q4_k_s = "gemma-3-4b-it-q4_k_s"
    case gemma_3_4b_it_q5_k_s = "gemma-3-4b-it-q5_k_s"
    case gemma_3_4b_it_q5_k_m = "gemma-3-4b-it-q5_k_m"

        // PHI-4 MINI INSTRUCT
    case qwen3_4b_instruct_2507_q3_k_m = "qwen3-4b-instruct-2507-q3_k_m"
    case qwen3_4b_instruct_2507_q3_k_s = "qwen3-4b-instruct-2507-q3_k_s"
    case qwen3_4b_instruct_2507_q4_k_m = "qwen3-4b-instruct-2507-q4_k_m"
    case qwen3_4b_instruct_2507_q4_k_s = "qwen3-4b-instruct-2507-q4_k_s"
    case qwen3_4b_instruct_2507_q5_k_s = "qwen3-4b-instruct-2507-q5_k_s"
    case qwen3_4b_instruct_2507_q5_k_m = "qwen3-4b-instruct-2507-q5_k_m"
    
    
    // Modelos anteriores
    case tinyllama_1_1b_chat_Q4_K_M = "tinyllama-1.1b-chat_Q4_K_M"
    case phi_2_Q4_K_M_1 = "phi-2_Q4_K_M"
    case gemma_2b_it_Q4_K_M = "gemma-2b-it_Q4_K_M"
    case stablelm_zephyr_3b_Q4_K_M = "stablelm-zephyr-3b_Q4_K_M"
    
    
    private var systemContextQuestionPrompt: String {
        return "You are a helpful assistant. Answer only with the final answer, using only the provided context. Do not explain or justify your answer."
    }
    private var systemSQLContextQuestionPrompt: String {
        return "You're a helpful assistant proficient in crafting SQL queries. The following command was used to create the table:"
    }
    
    
    var template: String {
            switch self {
            case .tinyllama_1_1b_chat_Q4_K_M:
                return "<|system|>\n{system}</s>\n<|user|>\n{prompt}</s>\n<|assistant|>"
            case .phi_2_Q4_K_M_1:
                return "{system}\nInstruct:{prompt}\nOutput:"
            case .gemma_2b_it_Q4_K_M:
                return "<start_of_turn>user\n{system}\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
            case .stablelm_zephyr_3b_Q4_K_M:
                return "<|user|>\n{system}\n{prompt}<|endoftext|>\n<|assistant|>\n"
            case 
                 .gemma_3_4b_it_q3_k_m,
                 .gemma_3_4b_it_q3_k_s,
                 .gemma_3_4b_it_q4_k_m,
                 .gemma_3_4b_it_q4_k_s,
                 .gemma_3_4b_it_q5_k_s,
                 .gemma_3_4b_it_q5_k_m:
                return """
        <start_of_turn>user
        {system}

        Context:
        {context}

        Question:
        {question}<end_of_turn>
        <start_of_turn>model
        """
            case
                 .qwen3_4b_instruct_2507_q3_k_m,
                 .qwen3_4b_instruct_2507_q3_k_s,
                 .qwen3_4b_instruct_2507_q4_k_m,
                 .qwen3_4b_instruct_2507_q4_k_s,
                 .qwen3_4b_instruct_2507_q5_k_s,
                 .qwen3_4b_instruct_2507_q5_k_m:
                return """
        <|im_start|>system
        {system}<|im_end|>
        <|im_start|>user
        Context:
        {context}
        
        Question:
        {question}<|im_end|>
        <|im_start|>assistant
        """

            }
        }
    func isQwenType() -> Bool {
            switch self {
            case
                 .qwen3_4b_instruct_2507_q3_k_m,
                 .qwen3_4b_instruct_2507_q4_k_m,
                 .qwen3_4b_instruct_2507_q3_k_s,
                 .qwen3_4b_instruct_2507_q4_k_s,
                 .qwen3_4b_instruct_2507_q5_k_s,
                 .qwen3_4b_instruct_2507_q5_k_m:
                return true
            default:
                return false
            }
        }

        func isGemmaType() -> Bool {
            switch self {
            case
                    .gemma_3_4b_it_q3_k_m,
                    .gemma_3_4b_it_q3_k_s,
                    .gemma_3_4b_it_q4_k_m,
                    .gemma_3_4b_it_q4_k_s,
                    .gemma_3_4b_it_q5_k_s,
                    .gemma_3_4b_it_q5_k_m,
                 .qwen3_4b_instruct_2507_q3_k_m,
                 .qwen3_4b_instruct_2507_q3_k_s,
                 .qwen3_4b_instruct_2507_q4_k_m,
                 .qwen3_4b_instruct_2507_q4_k_s,
                 .qwen3_4b_instruct_2507_q5_k_s,
                 .qwen3_4b_instruct_2507_q5_k_m:
                return true
            default:
                return false
            }
        }
    func templateBy(dataSetType: DatasetType, question: String, answer: String, context: String) -> String {
        
        switch dataSetType {
        case .hotpot_qa, .databricks_dolly:
            return self.template
                .replacingOccurrences(of: "{system}", with: systemContextQuestionPrompt)
                .replacingOccurrences(of: "{context}", with: context)
                .replacingOccurrences(of: "{question}", with: question)
        case .sql_create_context:
            return self.template
                .replacingOccurrences(of: "{system}", with: systemSQLContextQuestionPrompt)
                .replacingOccurrences(of: "{context}", with: context)
                .replacingOccurrences(of: "{question}", with: question)
            
        case .edinburgh_xsum:
            let aux = self.template
                .replacingOccurrences(of: "Context:", with: "")
                .replacingOccurrences(of: "{context}", with: "")
                .replacingOccurrences(of: "Question:", with: "")
                .replacingOccurrences(of: "{question}", with: "")
            
            let instructionPart = "Summarize the following article: " + question
                
            return aux.replacingOccurrences(of: "{system}", with: instructionPart) 
                
                
            
        default:
            return ""
        }
    }
}
