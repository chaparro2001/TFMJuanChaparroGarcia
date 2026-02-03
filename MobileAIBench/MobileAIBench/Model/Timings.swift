//
//  Timings.swift
//  MobileAIBench
//
//  Created by chaparro2001 on 9/11/25.
//

public struct Timings: Codable {

        // ms == milliseconds
    public var t_start_ms: Double  // absolute start time
    public var t_load_ms: Double   // time needed for loading the model
    public var t_p_eval_ms: Double // time needed for processing the prompt
    public var t_eval_ms: Double  // time needed for generating tokens

    public var n_p_eval: Int32   // number of prompt tokens
    public var n_eval: Int32     // number of generated tokens
    public var n_reused: Int32   // number of times a ggml compute graph had been reused



    public var t_sample_ms: Double // time needed for sampling in ms

    public var n_sample: Int32   // number of sampled tokens
    
    public var t_end_ms: Double
        


}
