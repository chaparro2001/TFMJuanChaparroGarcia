//
//  CPUMemoryUsage.swift
//  MobileAIBench
//
//  Created by chaparro2001 on 9/11/25.
//

import Foundation
public struct CPUMemoryUsage: Codable {
    let cpuUsage: Double
    let memoryUsage: Double
    let date: Date
}

public struct SysteUsage: Codable {
    var systemUsageRecords: [CPUMemoryUsage] = []
}
