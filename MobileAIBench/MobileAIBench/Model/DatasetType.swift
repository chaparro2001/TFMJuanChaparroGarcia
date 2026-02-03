//
//  DatasetType.swift
//  MobileAIBench
//
//  Created by chaparro2001 on 9/11/25.
//

public enum DatasetType: String, CaseIterable, Codable {
    
    static func getTestByIndex(index: Int) -> DatasetType {
        let allDatasets = DatasetType.allCases
        guard index >= 0 && index < allDatasets.count else {
            return DatasetType.allCases[0]
        }
        return allDatasets[index]
    }
    
    case hotpot_qa
    case sql_create_context
    case databricks_dolly
    case edinburgh_xsum
    case vqav2
    case scienceqa
    
    static var normal_tasks: [DatasetType] {
        return [.hotpot_qa, .databricks_dolly, .sql_create_context, .edinburgh_xsum]
    }
    
    static var multiModal_tasks: [DatasetType] {
        return [.vqav2, .scienceqa]
    }
}
