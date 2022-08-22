import math
mainConfig = {
    '1.0': {
      "version": '1.0',
      "threadsPerWarp": 32,
      "warpsPerMultiprocessor": 24,
      "threadsPerMultiprocessor": 768,
      "threadBlocksPerMultiprocessor": 8,
      "sharedMemoryPerMultiprocessor": 16384,
      "registerFileSize": 8192,
      "registerAllocationUnitSize": 256,
      "allocationGranularity": 'block',
      "maxRegistersPerThread": 124,
      "sharedMemoryAllocationUnitSize": 512,
      "warpAllocationGranularity": 2,
      "maxThreadBlockSize": 512
    },
    '1.1': {
      "version": '1.1',
      "threadsPerWarp": 32,
      "warpsPerMultiprocessor": 24,
      "threadsPerMultiprocessor": 768,
      "threadBlocksPerMultiprocessor": 8,
      "sharedMemoryPerMultiprocessor": 16384,
      "registerFileSize": 8192,
      "registerAllocationUnitSize": 256,
      "allocationGranularity": 'block',
      "maxRegistersPerThread": 124,
      "sharedMemoryAllocationUnitSize": 512,
      "warpAllocationGranularity": 2,
      "maxThreadBlockSize": 512
    },
    '1.2': {
      "version": '1.2',
      "threadsPerWarp": 32,
      "warpsPerMultiprocessor": 32,
      "threadsPerMultiprocessor": 1024,
      "threadBlocksPerMultiprocessor": 8,
      "sharedMemoryPerMultiprocessor": 16384,
      "registerFileSize": 16384,
      "registerAllocationUnitSize": 512,
      "allocationGranularity": 'block',
      "maxRegistersPerThread": 124,
      "sharedMemoryAllocationUnitSize": 512,
      "warpAllocationGranularity": 2,
      "maxThreadBlockSize": 512
    },
    '1.3': {
      "version": '1.3',
      "threadsPerWarp": 32,
      "warpsPerMultiprocessor": 32,
      "threadsPerMultiprocessor": 1024,
      "threadBlocksPerMultiprocessor": 8,
      "sharedMemoryPerMultiprocessor": 16384,
      "registerFileSize": 16384,
      "registerAllocationUnitSize": 512,
      "allocationGranularity": 'block',
      "maxRegistersPerThread": 124,
      "sharedMemoryAllocationUnitSize": 512,
      "warpAllocationGranularity": 2,
      "maxThreadBlockSize": 512
    },
    '2.0': {
      "version": '2.0',
      "threadsPerWarp": 32,
      "warpsPerMultiprocessor": 48,
      "threadsPerMultiprocessor": 1536,
      "threadBlocksPerMultiprocessor": 8,
      "sharedMemoryPerMultiprocessor": 49152,
      "registerFileSize": 32768,
      "registerAllocationUnitSize": 64,
      "allocationGranularity": 'warp',
      "maxRegistersPerThread": 63,
      "maxRegistersPerBlock": 32768,
      "sharedMemoryAllocationUnitSize": 128,
      "warpAllocationGranularity": 2,
      "maxThreadBlockSize": 1024
    },
    '2.1': {
      "version": '2.1',
      "threadsPerWarp": 32,
      "warpsPerMultiprocessor": 48,
      "threadsPerMultiprocessor": 1536,
      "threadBlocksPerMultiprocessor": 8,
      "sharedMemoryPerMultiprocessor": 49152,
      "registerFileSize": 32768,
      "registerAllocationUnitSize": 64,
      "allocationGranularity": 'warp',
      "maxRegistersPerThread": 63,
      "maxRegistersPerBlock": 32768,
      "sharedMemoryAllocationUnitSize": 128,
      "warpAllocationGranularity": 2,
      "maxThreadBlockSize": 1024
    },
    '3.0': {
      "version": '3.0',
      "threadsPerWarp": 32,
      "warpsPerMultiprocessor": 64,
      "threadsPerMultiprocessor": 2048,
      "threadBlocksPerMultiprocessor": 16,
      "sharedMemoryPerMultiprocessor": 49152,
      "registerFileSize": 65536,
      "registerAllocationUnitSize": 256,
      "allocationGranularity": 'warp',
      "maxRegistersPerThread": 63,
      "maxRegistersPerBlock": 65536,
      "sharedMemoryAllocationUnitSize": 256,
      "warpAllocationGranularity": 4,
      "maxThreadBlockSize": 1024
    },
    '3.2': {
      "version": '3.2',
      "threadsPerWarp": 32,
      "warpsPerMultiprocessor": 64,
      "threadsPerMultiprocessor": 2048,
      "threadBlocksPerMultiprocessor": 16,
      "sharedMemoryPerMultiprocessor": 49152,
      "registerFileSize": 65536,
      "registerAllocationUnitSize": 256,
      "allocationGranularity": 'warp',
      "maxRegistersPerThread": 255,
      "maxRegistersPerBlock": 65536,
      "sharedMemoryAllocationUnitSize": 256,
      "warpAllocationGranularity": 4,
      "maxThreadBlockSize": 1024
    },
    '3.5': {
      "version": '3.5',
      "threadsPerWarp": 32,
      "warpsPerMultiprocessor": 64,
      "threadsPerMultiprocessor": 2048,
      "threadBlocksPerMultiprocessor": 16,
      "sharedMemoryPerMultiprocessor": 49152,
      "registerFileSize": 65536,
      "registerAllocationUnitSize": 256,
      "allocationGranularity": 'warp',
      "maxRegistersPerThread": 255,
      "maxRegistersPerBlock": 65536,
      "sharedMemoryAllocationUnitSize": 256,
      "warpAllocationGranularity": 4,
      "maxThreadBlockSize": 1024
    },
    '3.7': {
      "version": '3.7',
      "threadsPerWarp": 32,
      "warpsPerMultiprocessor": 64,
      "threadsPerMultiprocessor": 2048,
      "threadBlocksPerMultiprocessor": 16,
      "sharedMemoryPerMultiprocessor": 114688,
      "registerFileSize": 131072,
      "registerAllocationUnitSize": 256,
      "allocationGranularity": 'warp',
      "maxRegistersPerThread": 255,
      "maxRegistersPerBlock": 65536,
      "sharedMemoryAllocationUnitSize": 256,
      "warpAllocationGranularity": 4,
      "maxThreadBlockSize": 1024
    },
    '5.0': {
      "version": '5.0',
      "threadsPerWarp": 32,
      "warpsPerMultiprocessor": 64,
      "threadsPerMultiprocessor": 2048,
      "threadBlocksPerMultiprocessor": 32,
      "sharedMemoryPerMultiprocessor": 65536,
      "registerFileSize": 65536,
      "registerAllocationUnitSize": 256,
      "allocationGranularity": 'warp',
      "maxRegistersPerThread": 255,
      "maxRegistersPerBlock": 65536,
      "sharedMemoryAllocationUnitSize": 256,
      "warpAllocationGranularity": 4,
      "maxThreadBlockSize": 1024
    },
    '5.2': {
      "version": '5.2',
      "threadsPerWarp": 32,
      "warpsPerMultiprocessor": 64,
      "threadsPerMultiprocessor": 2048,
      "threadBlocksPerMultiprocessor": 32,
      "sharedMemoryPerMultiprocessor": 98304,
      "registerFileSize": 65536,
      "registerAllocationUnitSize": 256,
      "allocationGranularity": 'warp',
      "maxRegistersPerThread": 255,
      "maxRegistersPerBlock": 32768,
      "sharedMemoryAllocationUnitSize": 256,
      "warpAllocationGranularity": 4,
      "maxThreadBlockSize": 1024
    },
    '5.3': {
      "version": '5.3',
      "threadsPerWarp": 32,
      "warpsPerMultiprocessor": 64,
      "threadsPerMultiprocessor": 2048,
      "threadBlocksPerMultiprocessor": 32,
      "sharedMemoryPerMultiprocessor": 65536,
      "registerFileSize": 65536,
      "registerAllocationUnitSize": 256,
      "allocationGranularity": 'warp',
      "maxRegistersPerThread": 255,
      "maxRegistersPerBlock": 32768,
      "sharedMemoryAllocationUnitSize": 256,
      "warpAllocationGranularity": 4,
      "maxThreadBlockSize": 1024
    },
    '6.0': {
      "version": '6.0',
      "threadsPerWarp": 32,
      "warpsPerMultiprocessor": 64,
      "threadsPerMultiprocessor": 2048,
      "threadBlocksPerMultiprocessor": 32,
      "sharedMemoryPerMultiprocessor": 65536,
      "registerFileSize": 65536,
      "registerAllocationUnitSize": 256,
      "allocationGranularity": 'warp',
      "maxRegistersPerThread": 255,
      "maxRegistersPerBlock": 65536,
      "sharedMemoryAllocationUnitSize": 256,
      "warpAllocationGranularity": 2,
      "maxThreadBlockSize": 1024
    },
    '6.1': {
      "version": '6.1',
      "threadsPerWarp": 32,
      "warpsPerMultiprocessor": 64,
      "threadsPerMultiprocessor": 2048,
      "threadBlocksPerMultiprocessor": 32,
      "sharedMemoryPerMultiprocessor": 98304,
      "registerFileSize": 65536,
      "registerAllocationUnitSize": 256,
      "allocationGranularity": 'warp',
      "maxRegistersPerThread": 255,
      "maxRegistersPerBlock": 65536,
      "sharedMemoryAllocationUnitSize": 256,
      "warpAllocationGranularity": 4,
      "maxThreadBlockSize": 1024
    },
    '6.2': {
      "version": '6.2',
      "threadsPerWarp": 32,
      "warpsPerMultiprocessor": 64,
      "threadsPerMultiprocessor": 2048,
      "threadBlocksPerMultiprocessor": 32,
      "sharedMemoryPerMultiprocessor": 65536,
      "registerFileSize": 65536,
      "registerAllocationUnitSize": 256,
      "allocationGranularity": 'warp',
      "maxRegistersPerThread": 255,
      "maxRegistersPerBlock": 65536,
      "sharedMemoryAllocationUnitSize": 256,
      "warpAllocationGranularity": 4,
      "maxThreadBlockSize": 1024
    },
    '7.0': {
      "version": '7.0',
      "threadsPerWarp": 32,
      "warpsPerMultiprocessor": 64,
      "threadsPerMultiprocessor": 2048,
      "threadBlocksPerMultiprocessor": 32,
      "sharedMemoryPerMultiprocessor": 98304,
      "registerFileSize": 65536,
      "registerAllocationUnitSize": 256,
      "allocationGranularity": 'warp',
      "maxRegistersPerThread": 255,
      "maxRegistersPerBlock": 65536,
      "sharedMemoryAllocationUnitSize": 256,
      "warpAllocationGranularity": 4,
      "maxThreadBlockSize": 1024
    },
    '7.5': {
      "version": '7.5',
      "threadsPerWarp": 32,
      "warpsPerMultiprocessor": 32,
      "threadsPerMultiprocessor": 1024,
      "threadBlocksPerMultiprocessor": 16,
      "sharedMemoryPerMultiprocessor": 65536,
      "registerFileSize": 65536,
      "registerAllocationUnitSize": 256,
      "allocationGranularity": 'warp',
      "maxRegistersPerThread": 255,
      "maxRegistersPerBlock": 65536,
      "sharedMemoryAllocationUnitSize": 256,
      "warpAllocationGranularity": 4,
      "maxThreadBlockSize": 1024
    },
    '8.0': {
      "version": '8.0',
      "threadsPerWarp": 32,
      "warpsPerMultiprocessor": 64,
      "threadsPerMultiprocessor": 2048,
      "threadBlocksPerMultiprocessor": 32,
      "sharedMemoryPerMultiprocessor": 167936,
      "registerFileSize": 65536,
      "registerAllocationUnitSize": 256,
      "allocationGranularity": 'warp',
      "maxRegistersPerThread": 255,
      "maxRegistersPerBlock": 65536,
      "sharedMemoryAllocationUnitSize": 128,
      "warpAllocationGranularity": 4,
      "maxThreadBlockSize": 1024
    },
    '8.6': {
      "version": '8.6',
      "threadsPerWarp": 32,
      "warpsPerMultiprocessor": 48,
      "threadsPerMultiprocessor": 1536,
      "threadBlocksPerMultiprocessor": 16,
      "sharedMemoryPerMultiprocessor": 102400,
      "registerFileSize": 65536,
      "registerAllocationUnitSize": 256,
      "allocationGranularity": 'warp',
      "maxRegistersPerThread": 255,
      "maxRegistersPerBlock": 65536,
      "sharedMemoryAllocationUnitSize": 128,
      "warpAllocationGranularity": 4,
      "maxThreadBlockSize": 1024
    }
}

cudaRuntimeUsedSharedMemory = {
'11.0': 1024,
'11.1': 1024
}

ceil = lambda a, b: math.ceil(a / b) * b
floor = lambda a, b: math.floor(a / b) * b

def calculateOccupancy(input):

# window.calculateOccupancy = function(input) {
# var activeThreadBlocksPerMultiprocessor, activeThreadsPerMultiprocessor, activeWarpsPerMultiprocessor, blockCudaRuntimeSharedMemory, blockRegisters, blockSharedMemory, blockWarps, config, occupancyOfMultiprocessor, output, registersPerWarp, threadBlocksPerMultiprocessorLimitedByRegistersPerMultiprocessor, threadBlocksPerMultiprocessorLimitedBySharedMemoryPerMultiprocessor, threadBlocksPerMultiprocessorLimitedByWarpsOrBlocksPerMultiprocessor, warpsPerMultiprocessorLimitedByRegisters;
    config = mainConfig[input["version"]]
    blockWarps = math.ceil(input["threadsPerBlock"] / config["threadsPerWarp"])
    registersPerWarp = ceil(input["registersPerThread"] * config["threadsPerWarp"], config["registerAllocationUnitSize"])
    blockRegisters = ceil(ceil(blockWarps, config["warpAllocationGranularity"]) * input["registersPerThread"] * config["threadsPerWarp"], config["registerAllocationUnitSize"])\
        if config["allocationGranularity"] == 'block' else registersPerWarp * blockWarps
    warpsPerMultiprocessorLimitedByRegisters = floor(config["maxRegistersPerBlock"] / registersPerWarp, config["warpAllocationGranularity"])
    blockCudaRuntimeSharedMemory = cudaRuntimeUsedSharedMemory[input["cudaVersion"]] if (float(input["version"]) >= 8) else 0
    blockSharedMemory = ceil(int(input["sharedMemoryPerBlock"]) + blockCudaRuntimeSharedMemory, config["sharedMemoryAllocationUnitSize"])
    threadBlocksPerMultiprocessorLimitedByWarpsOrBlocksPerMultiprocessor = min(config["threadBlocksPerMultiprocessor"], math.floor(config["warpsPerMultiprocessor"] / blockWarps))
    if input["registersPerThread"] > config["maxRegistersPerThread"]:
        threadBlocksPerMultiprocessorLimitedByRegistersPerMultiprocessor = 0
    elif input["registersPerThread"] > 0:
        if (config["allocationGranularity"] == 'block'):
            threadBlocksPerMultiprocessorLimitedByRegistersPerMultiprocessor = math.floor(config["registerFileSize"] / blockRegisters)
        else:
            threadBlocksPerMultiprocessorLimitedByRegistersPerMultiprocessor = math.floor(warpsPerMultiprocessorLimitedByRegisters / blockWarps) * math.floor(config["registerFileSize"] / config["maxRegistersPerBlock"])
    else:
        threadBlocksPerMultiprocessorLimitedByRegistersPerMultiprocessor = config["threadBlocksPerMultiprocessor"]
    

    if (input["sharedMemoryPerBlock"] > 0):
        threadBlocksPerMultiprocessorLimitedBySharedMemoryPerMultiprocessor = math.floor(config["sharedMemoryPerMultiprocessor"] / blockSharedMemory)
    else:
        threadBlocksPerMultiprocessorLimitedBySharedMemoryPerMultiprocessor = config["threadBlocksPerMultiprocessor"]
    activeThreadBlocksPerMultiprocessor = min(threadBlocksPerMultiprocessorLimitedByWarpsOrBlocksPerMultiprocessor,
     threadBlocksPerMultiprocessorLimitedByRegistersPerMultiprocessor, threadBlocksPerMultiprocessorLimitedBySharedMemoryPerMultiprocessor)
    activeWarpsPerMultiprocessor = activeThreadBlocksPerMultiprocessor* blockWarps
    activeThreadsPerMultiprocessor = input["threadsPerBlock"] * activeThreadBlocksPerMultiprocessor
    occupancyOfMultiprocessor = activeWarpsPerMultiprocessor / config["warpsPerMultiprocessor"]
    return occupancyOfMultiprocessor

if __name__=="__main__":
    threads_per_block = int(input("threads per block : "))
    registers_per_thread = int(input("registers per thread : "))
    shared_memory_per_block = int(input("shared memory per block : "))
    input = {
        "version": "7.0",
        "threadsPerBlock": threads_per_block,
        "registersPerThread": registers_per_thread,
        "cudaVersion": "11.1",
        "sharedMemoryPerBlock" : shared_memory_per_block
    }
    print(calculateOccupancy(input))