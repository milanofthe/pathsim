########################################################################################
##
##                   METHODS FOR SERIALIZATION OF PATHSIM OBJECTS
##                            (utils/serialization.py)
##
##                                Milan Rother 2025
##
########################################################################################

# IMPORTS ==============================================================================

import numpy as np

import importlib
import inspect
import base64
import types
import json
import dill
import sys


# SERIALIZATION ========================================================================

def serialize_callable(func):
    """Serialize a callable with priority for human-readable formats
    
    Parameters
    ----------
    func : callable
        function to serialize into dict

    Returns
    -------
    dict
        serialized function
    """
    
    #case 1: built-in function
    if isinstance(func, types.BuiltinFunctionType):
        return {
            "type": "builtin",
            "module": func.__module__,
            "name": func.__qualname__
        }
    
    #case 2: module-level function or class method from standard library
    if (hasattr(func, "__module__") and 
        (func.__module__ in sys.modules) and 
        not func.__module__.startswith('__main__')):
        
        try:
            #verify we can resolve reference
            module = importlib.import_module(func.__module__)
            obj = module
            for part in func.__qualname__.split('.'):
                obj = getattr(obj, part)
            
            #make sure we got the same function back
            if obj is func:  
                return {
                    "type": "reference",
                    "module": func.__module__,
                    "qualname": func.__qualname__
                }
        except (ImportError, AttributeError):
            pass  #fall through to next method
    
    #case 3: last resort -> use dill
    serialized = dill.dumps(
        func, 
        recurse=True, 
        byref=False,
        protocol=dill.HIGHEST_PROTOCOL
        )
    return {
        "type": "dill",
        "data": base64.b64encode(serialized).decode('ascii'),
        "name": getattr(func, "__name__", "unknown")
    }


def serialize_object(obj):
    """Serialize any object by capturing its module and class

    Parameters
    ----------
    obj : object
        object to serialize into dict

    Returns
    -------
    dict
        serialized object
    """
    
    #case 1: direct serialization
    try:
        json.dumps(obj)
        return obj

    #case 2: specific strategies
    except (TypeError, OverflowError):
        
        #get module and class info from the class object, not the instance
        if hasattr(obj, '__class__'):

            #get class info from the class itself
            cls = obj.__class__
            module_name = getattr(cls, '__module__', None)
            class_name = getattr(cls, '__name__', str(cls))
            
            #handle basic types with simple conversion methods
            if hasattr(obj, "tolist"):
                return {
                    "type": "object",
                    "__module__": module_name,
                    "__class__": class_name,
                    "data": obj.tolist()
                }

            elif hasattr(obj, "__list__"):
                return {
                    "type": "object",
                    "__module__": module_name,
                    "__class__": class_name,
                    "data": list(obj)
                }
        
    #case 3: last resort -> use dill
    serialized = dill.dumps(
        obj, 
        recurse=True, 
        byref=False,
        protocol=dill.HIGHEST_PROTOCOL
        )
    return {
        "type": "dill",
        "data": base64.b64encode(serialized).decode('ascii'),
        "name": getattr(obj, "__name__", "unknown")
    }


# DESERIALIZATION ======================================================================

def deserialize(data):
    """Deserialize an object from dictionary representation

    Parameters
    ----------
    data : dict
        dict to deserialize into object

    Returns
    -------
    object
        python object recovered from dict
    """
    
    #regular values and python objects
    if not isinstance(data, dict):
        return data

    #special types, builtin functions
    if data["type"] == "builtin":
        module = importlib.import_module(data["module"])
        names = data["name"].split('.')
        obj = module
        for name in names:
            obj = getattr(obj, name)
        return obj
    
    #functions with reference
    elif data["type"] == "reference":
        module = importlib.import_module(data["module"])
        names = data["qualname"].split('.')
        obj = module
        for name in names:
            obj = getattr(obj, name)
        return obj
    
    #dill
    elif data["type"] == "dill":
        return dill.loads(base64.b64decode(data["data"]))

    #other objects
    elif "__module__" in data and "__class__" in data:

        module_name, class_name = data["__module__"], data["__class__"]
                
        try:
            #try importing the module and class
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)

        except (ImportError, AttributeError) as E:
            raise E(f"<{module_name}.{class_name}> unrecoverable")

        if "data" in data:
            
            #get the data
            obj_data = data["data"]
        
            #objects with simple initialization from list
            if hasattr(cls, "from_list") and callable(cls.from_list):
                return cls.from_list(obj_data)
            
            #numpy-like arrays
            if module_name.startswith("numpy") and class_name.startswith("ndarray"):
                return np.array(obj_data)
            
            try:
                #everything else
                return cls(obj_data)

            except (AttributeError, ValueError) as E:              
                #data not recoverable
                raise E(f"<{module_name}.{data}> unrecoverable, {obj_data}")
    
    else:
        raise AttributeError(f"<{data}> unrecoverable")


# CLASS FOR AUTOMATIC SERIALIZATION CAPABILITIES =======================================

class Serializable:
    """Mixin that provides automatic serialization based on __init__ parameters 
    and loading/saving to json formatted readable files
    """
    
    def __str__(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=False)


    def save(self, path="", **metadata):
        """Save the dictionary representation of object to an external file
        
        Parameters
        ----------
        path : str
            filepath to save data to
        metadata : dict
            metadata for the object
        """
        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.to_dict(**metadata), file, indent=2, ensure_ascii=False)


    @classmethod
    def load(cls, path="", **kwargs):
        """Load and instantiate an object from an external file in json format
        
        Parameters
        ----------
        path : str
            filepath to load data from
        kwargs : dict
            additional kwargs for object reconstruction

        Returns
        -------
        out : obj
            reconstructed object from dict representation
        """
        with open(path, "r", encoding="utf-8") as file:
            return cls.from_dict(json.load(file), **kwargs)
        return None
        

    def to_dict(self, **metadata):
        """Convert object to dictionary representation

        Parameters
        ----------
        metadata : dict
            metadata for the object
        
        Returns
        -------
        data : dict
            representation of object
        """
        
        #get parameter names from __init__ signature
        signature = inspect.signature(self.__init__)
        param_names = [p for p in signature.parameters if p != "self"]

        #get current values of parameters
        params = {}        
        for name in param_names:
            
            if hasattr(self, name):
                
                value = getattr(self, name)

                #handle callable parameters
                if callable(value):
                    params[name] = serialize_callable(value)

                else:
                    params[name] = serialize_object(value)
            
        return {
            "id"       : id(self),
            "type"     : self.__class__.__name__,
            "metadata" : metadata,
            "params"   : params
        }
    

    @classmethod
    def from_dict(cls, data, **kwargs):
        """Create block instance from dictionary representation.

        Parameters
        ----------
        data : dict
            representation of object
        kwargs : dict
            additional kwargs for object reconstruction

        Returns
        -------
        out : obj
            reconstructed object from dict representation            
        """

        # Use the class specified in the data
        block_type = data.get("type")
        
        # Find the class in the module hierarchy
        target_cls = cls._find_class(block_type)
        
        # If this is already the target class, or we couldn't find the target
        if target_cls is None or target_cls == cls:

            #deserialize parameters
            params = {}
            for name, value in data["params"].items():
                params[name] = deserialize(value)

            #update optional kwargs
            for name, value in kwargs.items():
                params[name] = value
        
            #create the instance
            return cls(**params)
        
        else:
            #target class handle deserialization
            return target_cls.from_dict(data)


    @classmethod
    def _find_class(cls, class_name):
        """Find a class by name in the module hierarchy"""
        
        #first check if this is the class we're looking for
        if cls.__name__ == class_name:
            return cls
        
        #if not, check all subclasses recursively
        for subclass in cls.__subclasses__():
            if subclass.__name__ == class_name:
                return subclass
            
            #recursively check subclasses of this subclass
            found = subclass._find_class(class_name)
            if found:
                return found
        
        return None