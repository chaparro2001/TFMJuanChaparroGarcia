//
//  LibLlama.swift
//  LLMBench
//
//  Created by Tulika Awalgaonkar on 4/10/24.
//

import Foundation
import llama


enum LlamaError: Error {
    case couldNotInitializeContext
}

func llama_batch_clear(_ batch: inout llama_batch) {
    batch.n_tokens = 0
}

func llama_batch_add(_ batch: inout llama_batch, _ id: llama_token, _ pos: llama_pos, _ seq_ids: [llama_seq_id], _ logits: Bool) {
    batch.token   [Int(batch.n_tokens)] = id
    batch.pos     [Int(batch.n_tokens)] = pos
    batch.n_seq_id[Int(batch.n_tokens)] = Int32(seq_ids.count)
    for i in 0..<seq_ids.count {
        batch.seq_id[Int(batch.n_tokens)]![Int(i)] = seq_ids[i]
    }
    batch.logits  [Int(batch.n_tokens)] = logits ? 1 : 0

    batch.n_tokens += 1
}


actor LlamaContext {
    private var model: OpaquePointer
    private var context: OpaquePointer
    private var vocab: OpaquePointer
    private var batch: llama_batch
    private var tokens_list: [llama_token]
    private var sampling: UnsafeMutablePointer<llama_sampler>

    /// This variable is used to store temporarily invalid cchars
    private var temporary_invalid_cchars: [CChar]
    
    var final_output_string=""

    var n_len: Int32 = 1024
    var n_cur: Int32 = 0
    var n_start: Int32 = 0

    var n_decode: Int32 = 0
    
    public var is_done: Bool = false

    
    private var bosToken: llama_token
       private var eosToken: llama_token
       private var eotToken: llama_token  // End of Turn para Gemma
       private var nlToken: llama_token   // Newline
       private var infillToken: llama_token
    
    init(model: OpaquePointer, context: OpaquePointer) {
        self.model = model
        self.context = context
        self.tokens_list = []
        self.batch = llama_batch_init(1024, 0, 1)
        self.temporary_invalid_cchars = []
        let sparams = llama_sampler_chain_default_params()
        self.sampling = llama_sampler_chain_init(sparams)
        llama_sampler_chain_add(self.sampling, llama_sampler_init_temp(0.1))
        llama_sampler_chain_add(self.sampling, llama_sampler_init_dist(1234))
        
        llama_sampler_chain_add(self.sampling, llama_sampler_init_penalties(
            64,    // last_n
            1.1,   // repeat_penalty
            0.0,   // freq_penalty
            0.0    // presence_penalty
        ))

        //llama_sampler_chain_add(self.sampling, llama_sampler_init_top_k(40))
        //llama_sampler_chain_add(self.sampling, llama_sampler_init_top_p(0.9, 1))

        
        
        vocab = llama_model_get_vocab(model)
        
        
        self.bosToken = llama_vocab_bos(vocab)      // Begin of Sequence
                self.eosToken = llama_vocab_eos(vocab)      // End of Sequence
                self.eotToken = llama_vocab_eot(vocab)      // End of Turn (Gemma)
                self.nlToken = llama_vocab_nl(vocab)        // Newline
                self.infillToken = llama_vocab_mask(vocab)  // Infill
        print("🎫 Special Tokens:")
                    print("   BOS (Begin of Sequence): \(bosToken)")
                    print("   EOS (End of Sequence): \(eosToken)")
                    print("   EOT (End of Turn): \(eotToken)")
                    print("   NL (Newline): \(nlToken)")
                    print("   Infill: \(infillToken)")
    }
    
    deinit {
        llama_batch_free(batch)
        llama_free(context)
        llama_model_free(model)
        llama_backend_free()
    }

    static func create_context(path: String) throws -> LlamaContext {
        
        llama_backend_init()
        var model_params = llama_model_default_params()

#if targetEnvironment(simulator)
        model_params.n_gpu_layers = 0
        print("Running on simulator, force use n_gpu_layers = 0")
#endif
        //model_params.n_gpu_layers=45
        let model = llama_model_load_from_file(path, model_params)
        guard let model else {
            print("Could not load model at \(path)")
            throw LlamaError.couldNotInitializeContext
        }

        let n_threads = max(1, min(8, ProcessInfo.processInfo.processorCount - 2))
        print("Using \(n_threads) threads")

        var ctx_params = llama_context_default_params()
        //ctx_params.seed  = 12345678
        ctx_params.n_ctx = 5000
        ctx_params.n_threads       = Int32(UInt32(n_threads))
        ctx_params.n_threads_batch = Int32(UInt32(n_threads))
        
        let context = llama_init_from_model(model, ctx_params)//llama_new_context_with_model(model, ctx_params)
        guard let context else {
            print("Could not load context!")
            throw LlamaError.couldNotInitializeContext
        }

        return LlamaContext(model: model, context: context)
    }
    
    func getInfo() -> Timings{
        let context_data = llama_perf_context(context)
        let sampler_data = llama_perf_sampler(sampling)
        let t_end_ms = get_t_end_ms()
        
        let timings: Timings = Timings(t_start_ms: context_data.t_start_ms,
                                       t_load_ms: context_data.t_load_ms,
                                       t_p_eval_ms: context_data.t_p_eval_ms,
                                       t_eval_ms: context_data.t_eval_ms,
                                       n_p_eval: context_data.n_p_eval,
                                       n_eval: context_data.n_eval,
                                       n_reused: context_data.n_reused,
                                       t_sample_ms: sampler_data.t_sample_ms,
                                       n_sample: sampler_data.n_sample,
                                       t_end_ms: t_end_ms)
        
        return timings
    }
    
    func model_info() -> String {
        let result = UnsafeMutablePointer<Int8>.allocate(capacity: 256)
        result.initialize(repeating: Int8(0), count: 256)
        defer {
            result.deallocate()
        }

        // TODO: this is probably very stupid way to get the string from C

        let nChars = llama_model_desc(model, result, 256)
        let bufferPointer = UnsafeBufferPointer(start: result, count: Int(nChars))

        var SwiftString = ""
        for char in bufferPointer {
            SwiftString.append(Character(UnicodeScalar(UInt8(char))))
        }

        return SwiftString
    }

    func get_n_tokens() -> Int32 {
        return batch.n_tokens;
    }

    func completion_init(text: String, isGemma: Bool) {
        print("attempting to complete \"\(text)\"")

        tokens_list = tokenize(text: text, add_bos: true, isGemma: isGemma)
        temporary_invalid_cchars = []
        

        let n_ctx = llama_n_ctx(context)
        let n_kv_req = tokens_list.count + (Int(n_len) - tokens_list.count)

        //print("\n n_len = \(n_len), n_ctx = \(n_ctx), n_kv_req = \(n_kv_req)")

        if n_kv_req > n_ctx {
            print("error: n_kv_req > n_ctx, the required KV cache size is not big enough")
        }

//        for id in tokens_list {
//            print(String(cString: token_to_piece(token: id) + [0]))
//        }

        llama_batch_clear(&batch)

        for i1 in 0..<tokens_list.count {
            let i = Int(i1)
            llama_batch_add(&batch, tokens_list[i], Int32(i), [0], false)
        }
        batch.logits[Int(batch.n_tokens) - 1] = 1 // true

        if llama_decode(context, batch) != 0 {
            print("llama_decode() failed")
        }

        n_cur = batch.n_tokens
        n_start = batch.n_tokens
        //print("ncur \(n_cur)")
    }

    
    func completion_loop() -> (String, Bool) {
        var new_token_id: llama_token = 0

        new_token_id = llama_sampler_sample(sampling, context, batch.n_tokens - 1)
        llama_sampler_accept(sampling, new_token_id)
        
        if llama_vocab_is_eog(vocab, new_token_id) || n_cur == n_len {
            is_done = true
            let new_token_str = String(cString: temporary_invalid_cchars + [0])
            temporary_invalid_cchars.removeAll()
            return (new_token_str, true)
        }

        let new_token_cchars = token_to_piece(token: new_token_id)
        temporary_invalid_cchars.append(contentsOf: new_token_cchars)
        let new_token_str: String
        if let string = String(validatingUTF8: temporary_invalid_cchars + [0]) {
            temporary_invalid_cchars.removeAll()
            new_token_str = string
        } else if (0 ..< temporary_invalid_cchars.count).contains(where: {$0 != 0 && String(validatingUTF8: Array(temporary_invalid_cchars.suffix($0)) + [0]) != nil}) {
            // in this case, at least the suffix of the temporary_invalid_cchars can be interpreted as UTF8 string
            let string = String(cString: temporary_invalid_cchars + [0])
            temporary_invalid_cchars.removeAll()
            new_token_str = string
        } else {
            new_token_str = ""
        }
        print(new_token_str)
        // tokens_list.append(new_token_id)

        llama_batch_clear(&batch)
        llama_batch_add(&batch, new_token_id, n_cur, [0], true)

        n_decode += 1
        n_cur    += 1

        if llama_decode(context, batch) != 0 {
            print("failed to evaluate llama!")
        }

        return (new_token_str, false)
    }
    func llamaIsSeqEmpty(seq: Int32) -> Bool {
        let mem = llama_get_memory(context);
        return llama_memory_seq_pos_max(mem, seq) == -1;
    }

    func llamaIsMemoryEmpty() -> Bool{
        let mem = llama_get_memory(context);
        var nseq: Int32  = Int32(llama_n_seq_max(context));
        for s in 0..<Int32(nseq) {
             if llama_memory_seq_pos_max(mem, s) != -1 {
                 return false
             }
         }
        return true;
    }

    func clear() {
        tokens_list.removeAll()
        temporary_invalid_cchars.removeAll()
        llama_memory_clear(llama_get_memory(context), true)
        print("Cleared memory \(llamaIsMemoryEmpty())")
        llama_memory_seq_rm(llama_get_memory(context), 0, 0, -1)
        
    }

    private func tokenize(text: String, add_bos: Bool, isGemma: Bool) -> [llama_token] {
        let utf8Count = text.utf8.count
        let n_tokens = utf8Count + (add_bos ? 1 : 0) + 1
        let tokens = UnsafeMutablePointer<llama_token>.allocate(capacity: n_tokens)
        let tokenCount = llama_tokenize(vocab, text, Int32(utf8Count), tokens, Int32(n_tokens), add_bos, isGemma)
        
        var swiftTokens: [llama_token] = []
        for i in 0..<tokenCount {
            swiftTokens.append(tokens[Int(i)])
        }

        tokens.deallocate()

        return swiftTokens
    }

    /// - note: The result does not contain null-terminator
    private func token_to_piece(token: llama_token) -> [CChar] {
            let result = UnsafeMutablePointer<Int8>.allocate(capacity: 8)
            result.initialize(repeating: Int8(0), count: 8)
            defer {
                result.deallocate()
            }
            let nTokens = llama_token_to_piece(vocab, token, result, 8, 0,false)

            if nTokens < 0 {
                let newResult = UnsafeMutablePointer<Int8>.allocate(capacity: Int(-nTokens))
                newResult.initialize(repeating: Int8(0), count: Int(-nTokens))
                defer {
                    newResult.deallocate()
                }
                let nNewTokens = llama_token_to_piece(vocab, token, newResult, -nTokens, 0,false)
                let bufferPointer = UnsafeBufferPointer(start: newResult, count: Int(nNewTokens))
                return Array(bufferPointer)
            } else {
                let bufferPointer = UnsafeBufferPointer(start: result, count: Int(nTokens))
                return Array(bufferPointer)
            }
        }
}
