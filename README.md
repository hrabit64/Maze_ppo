# Maze PPO reinforce learning



## 필요 라이브러리

Tensorflow-gpu 2.3.1

tensorboardX

numpy

tkinter



## 사용법

main.py 

```python
if(__name__ == "__main__"):
    train = train_sys()
    _ = input("press any key to start")
    _ = train.run()
```

위 부분을 실행시키면 됩니다.



rtx 2060으로 학습시켰으며 320 에피소드 정도 소요되었습니다.

