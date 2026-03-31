import onnxruntime as ort

sess = ort.InferenceSession("/home/nutech/Desktop/Rebar-Counting-Product/backend/models/yolo_seg.onnx",
                            providers=["CPUExecutionProvider"])

print("INPUTS:")
for i in sess.get_inputs():
    print(i.name, i.shape, i.type)

print("OUTPUTS:")
for o in sess.get_outputs():
    print(o.name, o.shape, o.type)
