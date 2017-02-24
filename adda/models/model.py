models = {}

def register_model_fn(name):
    def decorator(fn):
        models[name] = fn
        return fn
    return decorator

def get_model_fn(name):
    return models[name]
