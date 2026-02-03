//
//  LlamaState.swift
//  LLMBench
//
//  Created by Tulika Awalgaonkar on 4/10/24.
//

import Foundation
import llama


import ProcessorKit


struct Question: Codable {
    let imageID: String
    let prompt: String
    
    enum CodingKeys: String, CodingKey {
        case imageID = "image_id"
        case prompt
    }
}

typealias Questions = [String: Question]

class LlamaState: ObservableObject {

    @Published var messageLog = ""
    private var llamaContext: LlamaContext?
    let NS_PER_S = 1_000_000_000.0
    private var monitorTask: Task<Void, Never>? = nil
    var systemUsageRecords: [CPUMemoryUsage] = []
    var allTestsResults: MobileAIBenchTests =  MobileAIBenchTests(tests: [])
    var testData: MobileAIBenchTest!
    init() {
        print("IN START")
        
    }
    
    
    func loadModel(modelUrl: URL?) throws {
        print("load model")
        
        if let modelUrl {
            print(modelUrl)
            Task { @MainActor in
                messageLog += "\nLoading model...\n"
            }
            llamaContext = try LlamaContext.create_context(path: modelUrl.path())
            Task { @MainActor in
                messageLog += "Loaded model \(modelUrl.lastPathComponent)\n"
                
            }
            

        } else {
            Task { @MainActor in
                messageLog += "Load a model from the list below\n"
            }
        }
    }
    
    func eval_model(model: String, dataset:DatasetType, model_name: GGUFModel, no_of_examples:Int, include_context:Bool) async{
        
        self.testData = MobileAIBenchTest(modelName: model_name.rawValue,
                                         results: [],
                                          taskType: dataset,
                                          numberOfExamples: no_of_examples,
                                         startedAt: Date(),
                                         endedAt: nil)
        guard let llamaContext else {
            return
        }
        //await print("Informacion del modelo \(llamaContext.model_info())")
        var SYS = get_SYS_prompt(dataset: dataset.rawValue, include_context: include_context)
        var context=""
        if include_context==true{
            context="with context"
        }
        else{
            context="without context"
        }
        if let fileURL = Bundle.main.url(forResource: dataset.rawValue, withExtension: "json", subdirectory: "datasets") {
            do {
                // Read the JSON data from the file
                let jsonData = try Data(contentsOf: fileURL)
                
                // Parse the JSON data
                if let jsonArray = try JSONSerialization.jsonObject(with: jsonData, options: []) as? [[String: Any]] {
                    // Iterate over each dictionary in the JSON array
                    var count = 0
                    var sumMetric1 = 0.0
                    var sumMetric2 = 0.0
                    for jsonDict in jsonArray {
                        // Extract question and answer from each dictionary
                        if count>=no_of_examples{
                            break
                        }
                        if var question = jsonDict["question"] as? String, let actual_answer = jsonDict["answer"] as? String {
                            // Do something with the question and answer
                            var prompt = ""
                            switch model_name {
                            case .tinyllama_1_1b_chat_Q4_K_M, .phi_2_Q4_K_M_1, .gemma_2b_it_Q4_K_M, .stablelm_zephyr_3b_Q4_K_M:
                                var new_sys=""
                                var new_ques=""
                                let dictValue = model_name.template
                                let context = jsonDict["context"] as? String ?? ""
                                (new_sys, new_ques) = get_prompt(dataset: dataset.rawValue, question: question, SYS: SYS, include_context: include_context, con: context)
                                let prompt1 = dictValue.replacingOccurrences(of: "{system}", with: new_sys)
                                prompt = prompt1.replacingOccurrences(of: "{prompt}", with: new_ques)
                                break
                            case
                                 .gemma_3_4b_it_q3_k_m,
                                 .gemma_3_4b_it_q3_k_s,
                                 .gemma_3_4b_it_q4_k_m,
                                 .gemma_3_4b_it_q4_k_s,
                                 .gemma_3_4b_it_q5_k_s,
                                 .gemma_3_4b_it_q5_k_m:
                                
                                prompt = model_name.templateBy(dataSetType: dataset,
                                                               question: question,
                                                               answer: actual_answer,
                                                               context: include_context ? (jsonDict["context"] as? String ?? "") : "")
                                
                               
                            case 
                                    .qwen3_4b_instruct_2507_q3_k_m,
                                    .qwen3_4b_instruct_2507_q3_k_s,
                                    .qwen3_4b_instruct_2507_q4_k_m,
                                    .qwen3_4b_instruct_2507_q4_k_s,
                                    .qwen3_4b_instruct_2507_q5_k_s,
                                    .qwen3_4b_instruct_2507_q5_k_m:
                                prompt = model_name.templateBy(dataSetType: dataset,
                                                               question: question,
                                                               answer: actual_answer,
                                                               context: include_context ? (jsonDict["context"] as? String ?? "") : "")
                               
                            }
                            var startItemTestDate = Date()
                            await llamaContext.completion_init(text: prompt, isGemma: model_name.isGemmaType())
                            var expected_answer=""
                            var metric1=0.0
                            var metric2=0.0
                            var result=""
                            var isdone=false
                                while isdone==false{
                                    (result, isdone) = await llamaContext.completion_loop()
                                    let suma: String = expected_answer + "\(result)"
                                    if isdone==true{
                                        expected_answer += "\(result)"
                                        break
                                    }
                                    expected_answer += "\(result)"
                                }

                            let llama_timings = await llamaContext.getInfo()
                            print(llama_timings)
                            print("--------------")
                            print("PREGUNTA: \(question)")
                            print("PREDICCION: \(expected_answer)")
                            print("RESPUESTA ORIGINAL: \(actual_answer)")
                            print("--------------")
                            (metric1,metric2) = task_specific_metric(dataset: dataset.rawValue,
                                                                   actual: actual_answer,
                                                                   predicted: expected_answer)
                            sumMetric1+=metric1
                            sumMetric2+=metric2
                            await llamaContext.clear()
                            var testItemData = MobileAIBenchTestResultPromt(prompt: prompt,
                                                                            output: expected_answer,
                                                                            gold_answer: actual_answer,
                                                                            timings: llama_timings,
                                                                            startedAt: startItemTestDate,
                                                                            endedAt: Date(),
                                                                            metric1: metric1,
                                                                            metric2: metric2)
                            testData.results.append(testItemData)
                            
                        }
                        count += 1
                        print(count)
                    }
                    self.stopMonitoring()
                    //print avg here
                    if count > 0 {
                        print(count)
                        let llama_timings = await llamaContext.getInfo()
                        print(llama_timings)
                        let total_time = llama_timings.t_end_ms-llama_timings.t_start_ms
                        let PromptTPS = 1e3 / llama_timings.t_p_eval_ms * Double(llama_timings.n_p_eval)
                        let EvalTPS = 1e3 / llama_timings.t_eval_ms * Double(llama_timings.n_eval)
                        let SampleTPS = 1e3 / llama_timings.t_sample_ms * Double(llama_timings.n_sample)
                        
                        let model_load_time = llama_timings.t_load_ms / 1000.0
                        let averageTotalTime = total_time / (Double(count)*1000.0)
                        let averageSampleTime = Double(llama_timings.t_sample_ms) / (Double(count)*1000.0)
                        let averagePromptTime = Double(llama_timings.t_p_eval_ms) / (Double(count)*1000.0)
                        let averagePromptTokens = Double(llama_timings.n_p_eval)/Double(count)
                        let averagePromptTokenPerSec=PromptTPS
                        let averageEvalTime=Double(llama_timings.t_eval_ms) / (Double(count)*1000.0)
                        let averageEvalTokens=Double(llama_timings.n_eval)/Double(count)
                        let averageEvalTokenPerSec=EvalTPS
                        let averageMetric1 = sumMetric1/Double(count)
                        let averageMetric2 = sumMetric2/Double(count)
                        let fstring = """
                            \nModel load time: \(model_load_time) sec
                            \nAverage values on \(model_name) for \(dataset) dataset(\(context)) \(count) examples:
                            Number of input tokens: \(averagePromptTokens)
                            Time to first token \(averagePromptTime) sec
                            Input tokens per sec: \(averagePromptTokenPerSec)
                            Sample time \(averageSampleTime) sec
                            Sample tokens per sec: \(SampleTPS)
                            Number of output tokens \(averageEvalTokens)
                            Output eval time \(averageEvalTime) sec
                            Output token per sec: \(averageEvalTokenPerSec)
                            Total time \(averageTotalTime) sec
                            """
                        let benchmarkMetrics = BenchmarkMetrics(model_load_time: model_load_time,
                                                                count: count,
                                                                averagePromptTokens: averagePromptTokens,
                                                                averagePromptTime: averagePromptTime,
                                                                averagePromptTokenPerSec: averagePromptTokenPerSec,
                                                                averageSampleTime: averageSampleTime,
                                                                sampleTPS: SampleTPS,
                                                                averageEvalTokens: averageEvalTokens,
                                                                averageEvalTime: averageEvalTime,
                                                                averageEvalTokenPerSec: averageEvalTokenPerSec,
                                                                averageTotalTime: averageTotalTime)
                        testData.benchmarkMetrics = benchmarkMetrics
                        testData.timings = llama_timings
                        
                        Task { @MainActor in
                            messageLog += fstring
                            messageLog += print_task_specific_metric(dataset: dataset.rawValue, metric1: averageMetric1, metric2: averageMetric2)
                        }
                    } else {
                        print("No valid data found.")
                        Task { @MainActor in
                            messageLog+="\nERROR: No valid data found."
                        }
                    }
                } else {
                    print("JSON data is not in the expected format.")
                    Task { @MainActor in
                        messageLog+="\nERROR: JSON data is not in the expected format."
                    }
                }
            } catch {
                print("Error reading JSON file: \(error)")
                Task { @MainActor in
                    messageLog+="\nERROR: Error reading JSON file"
                }
            }
        } else {
            print("JSON file not found.")
            Task { @MainActor in
                messageLog+="\nERROR: JSON file not found."
            }
        }
        testData.endedAt = Date()
        allTestsResults.tests.append(testData)
        
    }
    func bench_all(model: GGUFModel, task_name:DatasetType, examples:Int) async{
        Task { @MainActor in
            messageLog += "\t  ***RUNNING BENCHMARKING***\n"
        }
        let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let model_path = documentsURL.appendingPathComponent("\(model.rawValue).gguf")

        guard FileManager.default.fileExists(atPath: model_path.path) else {
            print("❌ File does not exist in Documents: \(model_path.path)")
            Task { @MainActor in
                messageLog += "\nERROR: Model not found in Documents. Make sure \(model.rawValue).gguf is copied to the app’s Documents directory."
            }
            return
        }
            self.allTestsResults = self.loadBenchTestsFromDocuments() ?? MobileAIBenchTests(tests: [])
            print("File exists in bundle")
            print(model_path)
            let model_url_name = model_path.lastPathComponent
            print(model_url_name)
            
            do {
                try loadModel(modelUrl: model_path)
                await eval_model(model: model_path.path,
                                dataset: task_name,
                                model_name: model,
                                 no_of_examples: examples,

                                include_context: true)
                saveBenchTestsToDocuments(allTestsResults)
            } catch {
                print("error: \(error)")
                messageLog += "\nEncountered unexpected ERROR: \(error.localizedDescription)"
            }
    }
    
    func loadBenchTestsFromDocuments(filename: String = "bench_results.json") -> MobileAIBenchTests? {
        do {
            // 1️⃣ Localiza la carpeta Documents del sandbox de la app
            let docsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let fileURL = docsURL.appendingPathComponent(filename)
            
            // 2️⃣ Verifica que el archivo exista
            guard FileManager.default.fileExists(atPath: fileURL.path) else {
                print("❌ No existe el archivo \(filename) en Documents")
                return nil
            }
            
            // 3️⃣ Lee los datos del archivo
            let data = try Data(contentsOf: fileURL)
            
            // 4️⃣ Decodifica usando JSONDecoder
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .secondsSince1970 // adapta si tus fechas usan otro formato
            let tests = try decoder.decode(MobileAIBenchTests.self, from: data)
            
            print("✅ Cargado desde Documents: \(filename) (\(tests.tests.count) tests)")
            return tests
        } catch {
            print("⚠️ Error cargando JSON desde Documents: \(error)")
            return nil
        }
    }

    // MARK: - Guardar a Documents (persistente)

    @discardableResult
    func saveBenchTestsToDocuments(_ tests: MobileAIBenchTests,
                                   filename: String = "bench_results.json") -> URL? {
        do {
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .secondsSince1970
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(tests)

            let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let fileURL = docs.appendingPathComponent(filename)
            try data.write(to: fileURL, options: .atomic)

            print("✅ Guardado en Documents: \(fileURL.path)")
            return fileURL
        } catch {
            print("⚠️ Error guardando JSON: \(error)")
            return nil
        }
    }
    
    func startMonitoring() {
        // Evita lanzar varios loops simultáneos
        guard monitorTask == nil else { return }
        systemUsageRecords = []
        monitorTask = Task.detached {[weak self] in
            guard let self = self else { return }
            while !Task.isCancelled {
                let cpu = CPU.appUsage()
                var memory = 0.0
                if let aux = Memory.appUsage() {
                    memory = Double(aux)
                }
                
                systemUsageRecords.append(
                    CPUMemoryUsage(cpuUsage: Double(cpu),
                                   memoryUsage: memory,
                                   date: Date())
                )
                
                // Espera medio segundo (0.5 s = 500_000_000 ns)
                try? await Task.sleep(nanoseconds: 500_000_000)
            }
        }
    }

    func stopMonitoring() {
        monitorTask?.cancel()
        monitorTask = nil
    }
}
