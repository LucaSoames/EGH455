import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork, ImgDetectionsBridge

device = dai.Device(dai.DeviceInfo(DEVICE)) if DEVICE else dai.Device()
platform = device.getPlatform()
img_frame_type = dai.ImgFrame.Type.BGR888i if platform.name == "RVC4" else dai.ImgFrame.Type.BGR888p
visualizer = dai.RemoteConnection(httpPort=8082)

with dai.Pipeline(device) as pipeline:
    cam = pipeline.create(dai.node.Camera).build()
    nn_archive = dai.NNArchive(MODEL_PATH)
    # Create the neural network node
    nn_with_parser = pipeline.create(ParsingNeuralNetwork).build(
        cam.requestOutput((512, 288), type=img_frame_type, fps=30), 
        nn_archive
    )

    # Bridge the detections to the visualizer
    label_encoding = {k: v for k, v in enumerate(nn_archive.getConfig().model.heads[0].metadata.classes)}
    bridge = pipeline.create(ImgDetectionsBridge).build(nn_with_parser.out)
    bridge.setLabelEncoding(label_encoding)

    # Configure the visualizer node
    visualizer.addTopic("Video", nn_with_parser.passthrough, "images")
    visualizer.addTopic("Detections", bridge.out, "detections")

    pipeline.start()
    visualizer.registerPipeline(pipeline)
 
    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key from the remote connection!")
            break