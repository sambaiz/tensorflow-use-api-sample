from mnist import Mnist

mnist = Mnist()
mnist.train(20000)
mnist.save("model.ckpt")
mnist.close()
print("done")
