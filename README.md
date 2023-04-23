# STUDY-Dog_Breed_Classification

> [CNN 강의](https://github.com/Chaewon-Leee/TIL/tree/main/DL/Ch.7) 수강 이후, Dog breed 이미지에 대해 classification 문제 수행

- 기간 : 23.02.12 ~ 23.02.23

### 목표

1. 자신만의 모델을 하나 build-up (MobileNet)
2. Generalization 기법들을 적용하여 성능올려보기 (VGG-16)
3. ResNet 기반으로 transfer-learning 성능 올리기, 성능 리포트 (resnet152)

- **Data info**

  - [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)

- **실행**

  - 실행 전, 다음과 같은 양식으로 `slack.json` 제작 필요
    - 작성자의 경우, slack으로 결과값을 전송하였지만, 다른 tool 사용시 `send_slack.py` & `slack_messenger.py` 삭제
    ```json
    // slack.json
    {
      "Slack": {
        "WEB_HOOK_URL": "WEB_HOOK_URL",
        "CHANNEL": "CHANNEL",
        "ACCESSED_TOKEN": "ACCESSED_TOKEN"
      }
    }
    ```

  - `train.py` 실행
  ```python
  python train.py -c config.json
  ```

- **After Seminar**
  | 회고 | 내용 |
  | :----------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | 성능 | - `resnet`의 경우 90%을 목표로 잡았으나 결국 도달하지 못함 <br>- 좀 더 실험이 필요해보임|
  | Weight Standardization | - 구현하고자 했으나 실패, 좀 더 이해가 필요함|
  | 오답노트 Error Analysis | - test case 확인하는 코드까진 제작하였으나, 오답노트 수행까진 시간 부족으로 인해 도전하지 못했음 <br> - 추후, 왜 해당 결과가 나오는지 개선할 수 있는 방향을 찾을 필요가 있음|

- **notion page**
  - [Dog breed](https://royal-tiger-88d.notion.site/Dog-breed-Classification-2f98bfb939814251b5770954236114df)

- **Additional Project**
  - [Check](https://github.com/Chaewon-Leee/TIL/tree/main/DL)
