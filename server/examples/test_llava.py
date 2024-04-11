from text_generation_server.pb import generate_pb2_grpc, generate_pb2
from text_generation_server.models.llava import LlavaLM, LlavaBatch

model = LlavaLM(model_id="liuhaotian/llava-v1.5-7b")
print(model)