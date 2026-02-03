//
//  MobileAIBenchApp.swift
//  MobileAIBench
//
//  Created by Tulika Awalgaonkar on 6/3/24.
//

import SwiftUI
import SwiftData
import UIKit

@main
struct MobileAIBenchApp: App {
    var sharedModelContainer: ModelContainer = {
        UIApplication.shared.isIdleTimerDisabled = true
        let schema = Schema([
            Item.self,
        ])
        let modelConfiguration = ModelConfiguration(schema: schema, isStoredInMemoryOnly: false)

        do {
            return try ModelContainer(for: schema, configurations: [modelConfiguration])
        } catch {
            fatalError("Could not create ModelContainer: \(error)")
        }
    }()

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .modelContainer(sharedModelContainer)
    }
}
