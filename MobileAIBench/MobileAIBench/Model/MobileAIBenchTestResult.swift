//
//  MobileAIBenchTestResult.swift
//  MobileAIBench
//
//  Created by chaparro2001 on 9/11/25.
//

import Foundation

public struct MobileAIBenchTestResultPromt: Codable {
    public var prompt: String
    public var output: String
    public var gold_answer: String
    public var timings: Timings
    public var startedAt: Date?
    public var endedAt: Date?
    public var metric1: Double?
    public var metric2: Double?
    
    mutating func cleanTexts() {
        self.prompt = ""
        self.output = ""
        self.gold_answer = ""
    }
}



public struct BenchmarkMetrics: Codable {
    // Tiempo total de carga del modelo (en segundos)
    public var model_load_time: Double
    
    // Número de ejemplos procesados en el benchmark
    public var count: Int
    
    // --- Métricas promedio del benchmark ---
    
    // Tokens de entrada promedio
    public var averagePromptTokens: Double
    
    // Tiempo medio hasta el primer token (segundos)
    public var averagePromptTime: Double
    
    // Velocidad de procesamiento de tokens de entrada (tokens/seg)
    public var averagePromptTokenPerSec: Double
    
    // Tiempo medio de muestreo de salida (segundos)
    public var averageSampleTime: Double
    
    // Tokens por segundo durante el muestreo
    public var sampleTPS: Double
    
    // Tokens de salida promedio
    public var averageEvalTokens: Double
    
    // Tiempo medio de evaluación de salida (segundos)
    public var averageEvalTime: Double
    
    // Tokens de salida por segundo durante la evaluación
    public var averageEvalTokenPerSec: Double
    
    // Tiempo total promedio (entrada + salida)
    public var averageTotalTime: Double
}


public struct MobileAIBenchTest: Codable, Identifiable {
    public var id = UUID()
    public var modelName: String
    public var results: [MobileAIBenchTestResultPromt]
    public var taskType: DatasetType
    public var numberOfExamples: Int
    
    public var startedAt: Date?
    public var endedAt: Date?
    public var timings: Timings?
    
    // Métricas agregadas del benchmark
    public var benchmarkMetrics: BenchmarkMetrics?
    //public var systemInfo: SysteUsage?
    
   
}

public struct MobileAIBenchTests: Codable {
    public var tests: [MobileAIBenchTest]
    
    
    func nextTest() -> (model: GGUFModel, dataset: DatasetType, numberOfExamples: Int)? {
        
        let listNumberOfExamples = [10, 50]
        
        for numberOfExamples in listNumberOfExamples {
            for model in GGUFModel.ggufModels {
                for dataset in [DatasetType.hotpot_qa, DatasetType.sql_create_context,  .edinburgh_xsum, .databricks_dolly,] {
                    if !tests.contains(where: { $0.modelName == model.rawValue && $0.taskType == dataset && $0.numberOfExamples == numberOfExamples }) {
                        return (model: model, dataset: dataset, numberOfExamples: numberOfExamples)
                    }
                }
            }
        }
        
        return nil
    }
        
}


