import paddle
from model.deeplab import Res_Deeplab

if __name__ == "__main__":
    x = paddle.randn([1, 3, 224, 224])
    model = Res_Deeplab(num_classes=19)
    try:
        input_spec = list(paddle.static.InputSpec.from_tensor(paddle.to_tensor(t)) for t in (x, ))
        paddle.jit.save(model, input_spec=input_spec, path="./model")
        print('[JIT] paddle.jit.save successed.')
        exit(0)
    except Exception as e:
        print('[JIT] paddle.jit.save failed.')
        raise e
