//
//  Untitled.swift
//  MobileAIBench
//
//  Created by chaparro2001 on 30/11/25.
//

//
//  Pruebas.swift
//  MobileAIBench
//
//  Created by chaparro2001 on 17/11/25.
//  Refactored by Assistant (Low Memory Settings & Batch Fix)
//

import Foundation
import llama

enum LlamaError: Error {
    case couldNotInitializeContext
    case modelNotFound
}

// Helpers batch

func llama_batch_clear(_ batch: inout llama_batch) {
    batch.n_tokens = 0
}

func llama_batch_add(_ batch: inout llama_batch,
                     _ id: llama_token,
                     _ pos: llama_pos,
                     _ seq_ids: [llama_seq_id],
                     _ logits: Bool) {
    batch.token   [Int(batch.n_tokens)] = id
    batch.pos     [Int(batch.n_tokens)] = pos
    batch.n_seq_id[Int(batch.n_tokens)] = Int32(seq_ids.count)

    for i in 0..<seq_ids.count {
        batch.seq_id[Int(batch.n_tokens)]![Int(i)] = seq_ids[i]
    }

    batch.logits  [Int(batch.n_tokens)] = logits ? 1 : 0
    batch.n_tokens += 1
}

// MARK: - LlamaContext

actor LlamaContext {
    private var model: OpaquePointer
    private var context: OpaquePointer
    private var vocab: OpaquePointer
    private var batch: llama_batch
    private var tokens_list: [llama_token]
    
    private var sampling: UnsafeMutablePointer<llama_sampler>

    private var temporary_invalid_cchars: [CChar]
    var final_output_string = ""

    var n_len: Int32 = 0
    var n_prompt: Int32 = 0
    var n_predict: Int32 = 128

    var n_cur: Int32 = 0
    var n_start: Int32 = 0
    var n_decode: Int32 = 0
    public var is_done: Bool = false
    
    // Configuración de capacidad del batch
    private let n_batch_capacity: Int32

    // Tokens especiales
    private var bosToken: llama_token
    private var eosToken: llama_token
    private var eotToken: llama_token
    private var nlToken: llama_token
    private var infillToken: llama_token
    
    let stopSequences = [
        "<|end|>",       // Fin normal de Phi
        "<|im_end|>",    // Fin alternativo
        "</s>",          // Fin de Llama/Mistral
        "<end_of_turn>", // Fin de Gemma
        "<|user|>",      // ALUCINACIÓN: El modelo intenta simular al usuario
        "<|system|>",    // ALUCINACIÓN: El modelo intenta reescribir reglas
        "User:",         // Variación común
        "Question:",      // Variación común
        "<|end_of_document|>"
    ]
    

    // MARK: - Init / Deinit

    init(model: OpaquePointer, context: OpaquePointer) {
        
        self.model = model
        self.context = context
        self.tokens_list = []
        
        // REDUCIDO: 512 es mucho más seguro para la RAM de un iPhone
        self.n_batch_capacity = 512
        self.batch = llama_batch_init(n_batch_capacity, 0, 1)
        
        self.temporary_invalid_cchars = []

        // Sampler
        let sparams = llama_sampler_chain_default_params()
        self.sampling = llama_sampler_chain_init(sparams)

        llama_sampler_chain_add(self.sampling, llama_sampler_init_penalties(64, 1.1, 0.0, 0.0))
        llama_sampler_chain_add(self.sampling, llama_sampler_init_temp(0.4))
        let seed = UInt32(llama_time_us() & 0xFFFFFFFF)
        llama_sampler_chain_add(self.sampling, llama_sampler_init_dist(seed))

        vocab = llama_model_get_vocab(model)

        self.bosToken    = llama_vocab_bos(vocab)
        self.eosToken    = llama_vocab_eos(vocab)
        self.eotToken    = llama_vocab_eot(vocab)
        self.nlToken     = llama_vocab_nl(vocab)
        self.infillToken = llama_vocab_mask(vocab)
        
        print("🎫 Batch capacity set to: \(n_batch_capacity)")
    }

    deinit {
        llama_batch_free(batch)
        llama_sampler_free(sampling)
        llama_free(context)
        llama_model_free(model)
        llama_backend_free()
    }

    // MARK: - Factory

    static func create_context(path: String) throws -> LlamaContext {
        llama_backend_init()
        var model_params = llama_model_default_params()

        #if targetEnvironment(simulator)
        model_params.n_gpu_layers = 0
        #endif

        let model = llama_model_load_from_file(path, model_params)
        guard let model else {
            print("Could not load model at \(path)")
            throw LlamaError.modelNotFound
        }

        let n_threads = max(1, min(8, ProcessInfo.processInfo.processorCount - 2))
        
        var ctx_params = llama_context_default_params()
        // REDUCIDO: 2048 tokens de contexto (ahorra memoria KV)
        ctx_params.n_ctx          = 2048
        
        // REDUCIDO: Batch de 512 (ahorra memoria Compute Buffer)
        ctx_params.n_batch        = 512
        
        ctx_params.n_ubatch = 64
        
        ctx_params.n_threads      = Int32(UInt32(n_threads))
        ctx_params.n_threads_batch = Int32(UInt32(n_threads))

        let context = llama_init_from_model(model, ctx_params)
        guard let context else {
            print("Could not load context!")
            throw LlamaError.couldNotInitializeContext
        }

        return LlamaContext(model: model, context: context)
    }

    // MARK: - Init de generación (Con Batching Seguro)

    func completion_init(text: String, isGemma: Bool) {
        print("attempting to complete \"\(text)\"")

        // 1. Tokenizar
        tokens_list = tokenize(text: text, add_bos: true, isGemma: isGemma)
        temporary_invalid_cchars.removeAll()
        final_output_string = ""
        is_done = false
        n_decode = 0

        let n_ctx = llama_n_ctx(context)
        n_prompt = Int32(tokens_list.count)

        // 2. Truncar si es necesario
        if n_prompt >= n_ctx {
            print("⚠️ Warning: Prompt larger than context window. Truncating.")
            let diff = Int(n_prompt) - Int(n_ctx) + Int(n_predict) + 10
            if diff < tokens_list.count {
                tokens_list.removeFirst(diff)
                n_prompt = Int32(tokens_list.count)
            }
        }
        n_len = min(n_prompt + n_predict, Int32(n_ctx))

        // 3. Procesar Prompt por Lotes
        llama_batch_clear(&batch)
        
        for i in 0..<tokens_list.count {
            // Verificar si es el último token GLOBAL del prompt
            let isLastToken = (i == tokens_list.count - 1)
            
            llama_batch_add(&batch, tokens_list[i], Int32(i), [0], isLastToken)
            
            // Si llenamos el batch, ejecutamos inferencia y limpiamos
            if batch.n_tokens == n_batch_capacity {
                if llama_decode(context, batch) != 0 {
                    print("❌ llama_decode() failed during prompt processing")
                    return
                }
                llama_batch_clear(&batch)
            }
        }
        
        // Procesar lo que haya quedado pendiente en el batch
        if batch.n_tokens > 0 {
            if llama_decode(context, batch) != 0 {
                print("❌ llama_decode() failed at end of prompt")
            }
        }

        n_cur   = n_prompt
        n_start = n_prompt
    }
    // En MobileAIBench/Pruebas.swift

        // MARK: - Bucle de generación
        func completion_loop() -> (String, Bool) {
            // 1) Samplear
            let new_token_id: llama_token = llama_sampler_sample(sampling, context, -1)
            llama_sampler_accept(sampling, new_token_id)

            // 2) Condición de parada (EOG o longitud)
            let isEog = llama_vocab_is_eog(vocab, new_token_id)
            if isEog || n_cur >= n_len {
                is_done = true
                let res = String(cString: temporary_invalid_cchars + [0])
                temporary_invalid_cchars.removeAll()
                // print("🔚 STOP: (EOG/Len) \(new_token_id)") // Debug opcional
                return (res, true)
            }

            // 3) Token -> String
            let new_token_str = convertTokenToString(token: new_token_id)

            // 4) COMPROBACIÓN DE PARADA BLINDADA
            if !new_token_str.isEmpty {
                print(new_token_str, terminator: "")
                final_output_string += new_token_str

                // LISTA DE PROHIBIDOS: Si aparece cualquiera de estos, CORTAMOS.
               
                // Usamos .contains para detectar la etiqueta aunque esté pegada a otra palabra o salto de línea
                for stopSeq in stopSequences {
                    if final_output_string.contains(stopSeq) {
                        is_done = true
                        
                        // Limpieza visual: Borramos la etiqueta del texto final para que el usuario no la vea
                        if let range = final_output_string.range(of: stopSeq) {
                            final_output_string = String(final_output_string[..<range.lowerBound])
                        }
                        // Borramos espacios en blanco sobrantes al final
                        final_output_string = final_output_string.trimmingCharacters(in: .whitespacesAndNewlines)
                        
                        print("\n🛑 STOP FORZADO: Se detectó '\(stopSeq)'")
                        return ("", true) // Devolvemos true para romper el bucle while
                    }
                }
            }

            // 5) Siguiente paso
            llama_batch_clear(&batch)
            llama_batch_add(&batch, new_token_id, n_cur, [0], true)

            n_decode += 1
            n_cur    += 1

            if llama_decode(context, batch) != 0 {
                print("failed to evaluate llama!")
                is_done = true
                return (new_token_str, true)
            }

            return (new_token_str, false)
        }
    // MARK: - Bucle de generación
    func completion_loopGemma() -> (String, Bool) {
        let new_token_id: llama_token = llama_sampler_sample(sampling, context, -1)
        llama_sampler_accept(sampling, new_token_id)

        let isEog = llama_vocab_is_eog(vocab, new_token_id)
        if isEog || n_cur >= n_len {
            is_done = true
            let res = String(cString: temporary_invalid_cchars + [0])
            temporary_invalid_cchars.removeAll()
            return (res, true)
        }

        let new_token_str = convertTokenToString(token: new_token_id)

        if !new_token_str.isEmpty {
            print(new_token_str, terminator: "")
            final_output_string += new_token_str

            let stopSequences = ["<|end|>", "<|im_end|>", "</s>", "<end_of_turn>"]
            for stopSeq in stopSequences {
                if final_output_string.hasSuffix(stopSeq) {
                    is_done = true
                    final_output_string.removeLast(stopSeq.count)
                    return ("", true)
                }
            }
        }

        llama_batch_clear(&batch)
        llama_batch_add(&batch, new_token_id, n_cur, [0], true)

        n_decode += 1
        n_cur    += 1

        if llama_decode(context, batch) != 0 {
            print("failed to evaluate llama!")
            is_done = true
            return (new_token_str, true)
        }

        return (new_token_str, false)
    }
    
    // MARK: - Utils
    
    func getInfo() -> Timings {
        let context_data = llama_perf_context(context)
        let sampler_data = llama_perf_sampler(sampling)
        let t_end_ms = get_t_end_ms()
        return Timings(
            t_start_ms: context_data.t_start_ms,
            t_load_ms: context_data.t_load_ms,
            t_p_eval_ms: context_data.t_p_eval_ms,
            t_eval_ms: context_data.t_eval_ms,
            n_p_eval: context_data.n_p_eval,
            n_eval: context_data.n_eval,
            n_reused: context_data.n_reused,
            t_sample_ms: sampler_data.t_sample_ms,
            n_sample: sampler_data.n_sample,
            t_end_ms: t_end_ms
        )
    }

    func model_info() -> String {
        let result = UnsafeMutablePointer<Int8>.allocate(capacity: 256)
        defer { result.deallocate() }
        let nChars = llama_model_desc(model, result, 256)
        let bufferPointer = UnsafeBufferPointer(start: result, count: Int(nChars))
        var swiftString = ""
        for char in bufferPointer {
            swiftString.append(Character(UnicodeScalar(UInt8(char))))
        }
        return swiftString
    }

    func get_n_tokens() -> Int32 { return batch.n_tokens }

    func clear() {
        tokens_list.removeAll()
        temporary_invalid_cchars.removeAll()
        final_output_string = ""
        let mem = llama_get_memory(context)
        llama_memory_clear(mem, true)
        llama_memory_seq_rm(mem, 0, 0, -1)
        n_cur = 0
        n_start = 0
        n_len = 0
        n_prompt = 0
        n_decode = 0
        is_done = false
        print("Cleared memory")
    }
    
    private func tokenize(text: String, add_bos: Bool, isGemma: Bool) -> [llama_token] {
        let utf8Count = text.utf8.count
        let n_tokens = utf8Count + (add_bos ? 1 : 0) + 1
        let tokens = UnsafeMutablePointer<llama_token>.allocate(capacity: n_tokens)
        defer { tokens.deallocate() }
        let tokenCount = llama_tokenize(vocab, text, Int32(utf8Count), tokens, Int32(n_tokens), add_bos, isGemma)
        var swiftTokens: [llama_token] = []
        if tokenCount > 0 {
            for i in 0..<tokenCount { swiftTokens.append(tokens[Int(i)]) }
        }
        return swiftTokens
    }

    private func convertTokenToString(token: llama_token) -> String {
        let new_token_cchars = token_to_piece(token: token)
        temporary_invalid_cchars.append(contentsOf: new_token_cchars)
        if let string = String(validatingUTF8: temporary_invalid_cchars + [0]) {
            temporary_invalid_cchars.removeAll()
            return string
        }
        return ""
    }

    private func token_to_piece(token: llama_token) -> [CChar] {
        let initialCap = 8
        let result = UnsafeMutablePointer<Int8>.allocate(capacity: initialCap)
        result.initialize(repeating: Int8(0), count: initialCap)
        defer { result.deallocate() }
        var nTokens = llama_token_to_piece(vocab, token, result, Int32(initialCap), 0, false)
        if nTokens < 0 {
            let needed = Int(-nTokens)
            let newResult = UnsafeMutablePointer<Int8>.allocate(capacity: needed)
            newResult.initialize(repeating: Int8(0), count: needed)
            defer { newResult.deallocate() }
            nTokens = llama_token_to_piece(vocab, token, newResult, -nTokens, 0, false)
            let bufferPointer = UnsafeBufferPointer(start: newResult, count: Int(nTokens))
            return Array(bufferPointer)
        } else {
            let bufferPointer = UnsafeBufferPointer(start: result, count: Int(nTokens))
            return Array(bufferPointer)
        }
    }
    
    // Stub
    func llamaIsMemoryEmpty() -> Bool { return true }
}
