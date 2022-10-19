from misc.log_utils import log, dict_to_string
from model import peopleflow, pipeline
# from model import centertrack


def pipelineFactory(data_spec):
    log.info(f"Building Pipeline")

    people_flow = peopleflow.PeopleFlowProb(data_spec)
    full_pipeline = pipeline.MultiViewPipeline(people_flow)

    return full_pipeline