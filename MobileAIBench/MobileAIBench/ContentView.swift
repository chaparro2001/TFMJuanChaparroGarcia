//
//  ContentView.swift
//  MobileAIBench
//
//  Created by Tulika Awalgaonkar on 6/3/24.
//

import SwiftUI
import SwiftData





@MainActor
final class LlamaStateModel: ObservableObject {
    @Published var messageLog: String = ""
    var llamaState = LlamaState()
    func bench_all(model: GGUFModel, task_name: DatasetType, examples: Int) async {
        // Implementación real aquí
        await llamaState.bench_all(model: model, task_name: task_name, examples: examples)
        messageLog += llamaState.messageLog
    }
    
    
    
    
    
}

// ======= CONTENT VIEW =======

struct ContentView: View {
    @StateObject var llamaState = LlamaStateModel()

    @State private var selectedModel: GGUFModel = GGUFModel.getModelByIndex(index: 0)
    @State private var selectedTask: DatasetType = DatasetType.getTestByIndex(index: 0)
    @State private var selectedExample: Int = 50

    private let models = GGUFModel.allCases
    private let text_tasks = DatasetType.normal_tasks          // [DatasetType]
    private let MM_tasks   = DatasetType.multiModal_tasks      // [DatasetType]
    private let text_examples = [10, 50, 100]
    private let MM_examples   = [10, 25, 50]
    
    @State private var showResults = false
    @State private var processing: Bool = false
    
    @State private var title: String = "Procesando..."

    // Ahora tasks es [DatasetType], no [String]
    private var tasks: [DatasetType] {
        switch selectedModel {
        case .tinyllama_1_1b_chat_Q4_K_M:
            return MM_tasks
        default:
            return text_tasks
        }
    }

    private var examples: [Int] {
        switch selectedModel {
        case .tinyllama_1_1b_chat_Q4_K_M:
            return MM_examples
        default:
            return text_examples
        }
    }
    var body: some View {
        NavigationStack {
            // 👉 NavigationLink programático
                        NavigationLink(
                            destination: BenchResultsView(
                                viewModel: BenchResultsViewModel(
                                    tests: llamaState.llamaState.allTestsResults.tests
                                )
                            ),
                            isActive: $showResults
                        ) {
                            EmptyView()
                        }
            VStack(spacing: 20) {
                Text("MobileAIBench")
                    .font(.title)
                    .fontWeight(.bold)
                    .padding(.top, 20)
                if processing {
                    Text(self.title)
                        .font(.title)
                        .fontWeight(.bold)
                        .padding(.top, 20)
                        }
                
                // ======= Controles =======
                VStack(spacing: 5) {
                    // Models Dropdown
                    VStack {
                        Text("Select Model")
                            .font(.headline)
                        
                        Picker("Select Model", selection: $selectedModel) {
                            ForEach(models, id: \.self) { model in
                                Text(model.rawValue).tag(model)
                            }
                        }
                        .pickerStyle(MenuPickerStyle())
                        .frame(maxWidth: .infinity)
                        .onChange(of: selectedModel) { newValue in
                            updateDefaultTask(for: newValue)
                            // Garantiza que la selección es válida dentro de la nueva lista
                            if let first = tasks.first {
                                selectedTask = first
                            }
                            // Opcional: al cambiar modelo, reajusta ejemplo si procede
                            if let firstEx = examples.first {
                                //selectedExample = firstEx
                            }
                        }
                    }
                    
                    // Task Dropdown
                    VStack {
                        Text("Select Task")
                            .font(.headline)
                        
                        Picker("Select Task", selection: $selectedTask) {
                            ForEach(tasks, id: \.self) { task in
                                Text(task.rawValue).tag(task) // tag del MISMO TIPO que selection
                            }
                        }
                        .pickerStyle(MenuPickerStyle())
                        .frame(maxWidth: .infinity)
                        .disabled(tasks.isEmpty)
                    }
                    
                    // Examples Dropdown
                    VStack {
                        Text("Select Example")
                            .font(.headline)
                        
                        Picker("Select Example", selection: $selectedExample) {
                            ForEach(examples, id: \.self) { example in
                                Text("\(example)").tag(example)
                            }
                        }
                        .pickerStyle(MenuPickerStyle())
                        .frame(maxWidth: .infinity)
                        .disabled(examples.isEmpty)
                    }
                }
                .padding()
                .background(Color(.systemGray6))
                .cornerRadius(10)
                .shadow(radius: 5)
                
                // ======= Log =======
                ScrollView(.vertical, showsIndicators: true) {
                    Text(llamaState.messageLog)
                        .font(.system(size: 12))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                        .onTapGesture {
#if os(iOS)
                            UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder),
                                                            to: nil, from: nil, for: nil)
#endif
                        }
                }
                
                // ======= Botón =======
                if !processing {
                    HStack {
                        Button(action: { submitData() }) {
                            Text("Run")
                                .padding()
                                .background(Color.blue)
                                .foregroundColor(.white)
                                .cornerRadius(8)
                        }
                        .disabled(examples.isEmpty)
                        
                        Button(action: { showResults = true  }
                                        
                                ) {
                            Text("Share Results")
                                .padding()
                                .background(Color.red)
                                .foregroundColor(.white)
                                .cornerRadius(8)
                        }
                        .disabled(examples.isEmpty)
                        
                        Button(action: { checkAllModelsExist()  }
                                        
                                ) {
                            Text("comprobar todos los modelos")
                                .padding()
                                .background(Color.red)
                                .foregroundColor(.white)
                                .cornerRadius(8)
                        }
                        .disabled(examples.isEmpty)
                        
                        Button(action: { nextTest()  }
                                        
                                ) {
                            Text("Test Automatico")
                                .padding()
                                .background(Color.green)
                                .foregroundColor(.white)
                                .cornerRadius(8)
                        }
                        .disabled(examples.isEmpty)
                        
                        
                    }
                }
            }
            .padding()
        }
        .navigationTitle("MobileAIBench")
        
    }

    private func updateDefaultTask(for model: GGUFModel) {
        switch model {
        case .tinyllama_1_1b_chat_Q4_K_M:
            selectedTask = .vqav2
        default:
            selectedTask = .hotpot_qa
        }
    }

    private func submitData() {
        Task { @MainActor in
            self.processing = true
            self.title = "Processing \(selectedModel.rawValue) on \(selectedTask.rawValue) with \(selectedExample) examples..."
            await llamaState.bench_all(model: selectedModel,
                                       task_name: selectedTask,
                                       examples: selectedExample)
            self.processing = false
        }
    }
    
    func checkAllModelsExist()  {
        var results: [GGUFModel: Bool] = [:]

        let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!

        for model in GGUFModel.allCases {
            let fileURL = documentsURL.appendingPathComponent(model.rawValue + ".gguf")

            let exists = FileManager.default.fileExists(atPath: fileURL.path)
            results[model] = exists

            if exists {
                print("✅ \(model.rawValue).gguf encontrado")
            } else {
                print("❌ FALTA \(model.rawValue).gguf")
            }
        }
    }
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

    func nextTest(filename: String = "bench_results.json") {
        do {
            // 1️⃣ Localiza la carpeta Documents del sandbox de la app
            let docsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let fileURL = docsURL.appendingPathComponent(filename)
            
            // 2️⃣ Verifica que el archivo exista
            guard FileManager.default.fileExists(atPath: fileURL.path) else {
                print("❌ No existe el archivo \(filename) en Documents")
                saveBenchTestsToDocuments(MobileAIBenchTests(tests: []), filename: filename)
                return
            }
            
            // 3️⃣ Lee los datos del archivo
            let data = try Data(contentsOf: fileURL)
            
            // 4️⃣ Decodifica usando JSONDecoder
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .secondsSince1970 // adapta si tus fechas usan otro formato
            let tests = try decoder.decode(MobileAIBenchTests.self, from: data)
            
            guard let nextText = tests.nextTest() else {
                print("✅ Todos los tests completados.")
                return
            }
            Task { @MainActor in
                self.processing = true
                self.title = "Processing \(nextText.model.rawValue) on \(nextText.dataset.rawValue) with \(nextText.numberOfExamples) examples..."
                await llamaState.bench_all(model: nextText.model,
                                           task_name: nextText.dataset,
                                           examples: nextText.numberOfExamples)
                self.processing = false
            }
            
            
            print("✅ Cargado desde Documents: \(filename) (\(tests.tests.count) tests)")
            
            
            
        } catch {
            print("⚠️ Error cargando JSON desde Documents: \(error)")
            
        }
    }
    
    
}




class BenchResultsViewModel: ObservableObject {
    @Published var tests: [MobileAIBenchTest] = []
    
    init(tests: [MobileAIBenchTest] = []) {
        self.tests = tests
        
        
        if tests.isEmpty {
            self.tests = self.loadBenchTestsFromDocuments()?.tests ?? []
        }
    }
    
    func delete(at offsets: IndexSet) {
            tests.remove(atOffsets: offsets)
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

    
    func saveBenchTestsToDocuments(filename: String = "bench_results.json") -> URL? {
        do {
            let test = MobileAIBenchTests(tests: tests)
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .secondsSince1970
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(test)

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
    
}

struct BenchResultsView: View {
    @ObservedObject var viewModel: BenchResultsViewModel
    
    var body: some View {
        List {
            ForEach(viewModel.tests) { test in
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(test.modelName)
                            .font(.headline)
                        
                        Text("Task: \(test.taskType.rawValue)")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                    
                    Spacer()
                    
                    Text("\(test.numberOfExamples)")
                        .font(.title3)
                        .bold()
                }
                .padding(.vertical, 8)
            }
            .onDelete { indexSet in
                viewModel.delete(at: indexSet)
            }
        }
        .navigationTitle("Benchmarks")
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                Button("Guardar") {
                    viewModel.saveBenchTestsToDocuments()
                }
            }
        }
    }
}
