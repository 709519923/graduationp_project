
'''

自定义信号

pyqtSignal()
用这个函数自定义信号

'''

from PyQt5.QtCore import *
class MyTypeSignal(QObject):
    # 定义一个信号
    sendmsg = pyqtSignal(object)

    # 发送3个参数的信号
    sendmsg1 = pyqtSignal(str,int,int)

    def run(self):
        self.sendmsg.emit('Hello PyQt5')

    def run1(self):
        self.sendmsg1.emit("hello",3,4)


class MySlot(QObject):
    def get(self,msg):
        print("信息：" + msg)A
    def get1(self,msg,a,b):
        print(msg)
        print(a+b)


if __name__ == '__main__':
    send = MyTypeSignal() #信号类
    slot = MySlot() #槽类

    send.sendmsg.connect(slot.get) #信号实例连接槽的某个方法
    send.sendmsg1.connect(slot.get1) #信号实例连接槽的某个方法

    #连接之后，通过run函数发送参数给槽函数，并触发槽函数
    send.run() 
    send.run1()
    #失去连接
    send.sendmsg.disconnect(slot.get)
    send.run()