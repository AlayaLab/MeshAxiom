"""MeshAxiom's lightweight Python interface. GPU libraries load in workers."""
__version__ = "0.2.0.dev1"

def __getattr__(name):
    if name == "MeshAxiomPipeline":
        from .pipeline import MeshAxiomPipeline
        return MeshAxiomPipeline
    raise AttributeError(name)

