//
//  Memory.swift
//  MobileAIBench
//
//  Created by chaparro2001 on 8/11/25.
//

import Combine

 enum MemoryMonitorState {
    case started
    case paused
}


class MemoryUsageCustom {
    
    private var displayLink: CADisplayLink!

    var state = MemoryMonitorState.paused
    
    let subject = PassthroughSubject<String, Never>()

    
    private static var sharedInstance: MemoryUsageCustom!
    
    public class func shared() -> MemoryUsageCustom {
        if self.sharedInstance == nil {
            self.sharedInstance = MemoryUsageCustom()
        }
        return self.sharedInstance
    }
    
    private init() {
        self.configureDisplayLink()

    }
    
    func startMemoryMonitor() {
        
        if self.state == .started {
            return
        }
        
        self.state = .started
        self.start()
    }
    
    func stopMemoryMonitor() {
        self.state = .paused
        self.pause()
    }
    

    //--------------------------------------------------------------------------------
    
    //MARK:- Display Link
    
    //--------------------------------------------------------------------------------

 
    func configureDisplayLink() {
        self.displayLink = CADisplayLink(target: self, selector: #selector(displayLinkAction(displayLink:)))
        self.displayLink.isPaused = true
        self.displayLink?.add(to: .current, forMode: .common)
    }
    
    private func start() {
        self.displayLink?.isPaused = false
    }
    
    /// Pauses performance monitoring.
    private func pause() {
        self.displayLink?.isPaused = true
    }
    
    @objc func displayLinkAction(displayLink: CADisplayLink) {
        let memoryUsage = self.memoryUsage()
        
        let bytesInMegabyte = 1024.0 * 1024.0
        let usedMemory = Double(memoryUsage.used) / bytesInMegabyte
        let totalMemory = Double(memoryUsage.total) / bytesInMegabyte
        let memory = String(format: "%.1f of %.0f MB used", usedMemory, totalMemory)

     //   self.memoryString = memory
        subject.send(memory)
    }
    
    func memoryUsage() -> (used: UInt64, total: UInt64) {
        var taskInfo = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info>.size) / 4
        let result: kern_return_t = withUnsafeMutablePointer(to: &taskInfo) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        
        var used: UInt64 = 0
        if result == KERN_SUCCESS {
            used = UInt64(taskInfo.phys_footprint)
        }
        
        let total = ProcessInfo.processInfo.physicalMemory
        return (used, total)
    }

}
