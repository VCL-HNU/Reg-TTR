

def getModel(opt):
    model_name = opt['model']
    if model_name == 'UniGradICON':
        import sys
        sys.path.insert(0, '/data2/cl/projects/SAMIR/models/uniGradICON/uniGradICON/src')
        from unigradicon import get_unigradicon

        print("✓ Loading UniGradICON from local source...")
        model = get_unigradicon()
        return model

