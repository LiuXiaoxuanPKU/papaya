from torch._utils import _get_device_index
from typing import List, Optional, Tuple, Union, Any
from torch.types import Device
import numpy as np
from multiprocessing import Process, Value, Manager
import subprocess as sp
import time

class gpuUtilization:
    def __init__(self,device: Optional[Union[Device, int]] = None):
        self.__manager = Manager()
        self.__device = device
        self.__n = 0
        self.samples = self.__manager.list()
    def reset(self,device: Optional[Union[Device, int]] = None):
        self.__device = device
        self.__n = 0
        self.samples = self.__manager.list()
    def get(self):
        # try: 
        #     ret = self.__utilization(self.__device)
        # except:
        ret = gpuUtilization.get_gpu_util()[0]
        return ret
        # return ret
    def profile(self, start_n = 0):
        self.__n+=1
        if self.__n>start_n: 
            ret = self.get()
            print("Get utilization {}%.".format(ret))
            self.samples.append(ret)
    def getAvg(self):
        return np.average(self.samples)
    def __utilization(self,device: Optional[Union[Device, int]] = None) -> int:
        try:
            import pynvml  # type: ignore[import]
        except ModuleNotFoundError:
            raise ModuleNotFoundError("pynvml module not found, please install pynvml")
        from pynvml import NVMLError_DriverNotLoaded
        try:
            pynvml.nvmlInit()
        except NVMLError_DriverNotLoaded:
            raise RuntimeError("cuda driver can't be loaded, is cuda enabled?")
        device = _get_device_index(device, optional=True)
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        return pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
    @staticmethod
    def get_gpu_util():
        output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
        ACCEPTABLE_AVAILABLE_MEMORY = 1024
        COMMAND = "nvidia-smi --query-gpu=utilization.gpu --format=csv"
        try:
            util_info = output_to_list(sp.check_output(COMMAND.split(),stderr=sp.STDOUT))[1:]
        except sp.CalledProcessError as e:
            raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
        util_values = [int(x.split()[0]) for i, x in enumerate(util_info)]
        return util_values

class UtilizationContext():
    def __init__(self,profile = False, gpu_utilization = None):
        self.__profile = profile
        self.__gpu_utilization = gpu_utilization
        if self.__profile and self.__gpu_utilization is None: self.__gpu_utilization = gpuUtilization()
        if self.__profile:
            self.__run_flag,self.__profile_flag = Value('b',0),Value('b', 0)
            self.__proc = Process(target=UtilizationContext.log_utilization, args=(self.__run_flag, self.__profile_flag,self.__gpu_utilization))
         
    def __enter__(self):
        if self.__profile:
            self.__run_flag.value = 1
            self.__proc.start()
        return self
     
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.__profile: 
            print("Destorying utilization profile process...")
            self.__run_flag.value = 0
            self.__proc.join()
        
    def getAvg(self):
        if not self.enabled: return None
        return self.__gpu_utilization.getAvg()
    
    def startProfile(self):
        self.__profile_flag.value = 1
        
    def endProfile(self):
        self.__profile_flag.value = 0
    
    @property
    def enabled(self): return self.__profile==True
    
    @property
    def profile(self): return self.__profile
    
    @staticmethod
    def log_utilization(run_flag,prof_flag,gpu):
        while run_flag.value==1:
            if prof_flag.value==1:
                gpu.profile(start_n = 2)
            time.sleep(0.18)
        
        
class UtilizationTrainContext():
    def __init__(self, context):
        self.__context = context
         
    def __enter__(self):
        if self.__context.profile: self.__context.startProfile()
        return self
     
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.__context.profile: self.__context.endProfile()
        

if __name__=="__main__": print(gpuUtilization().get())
