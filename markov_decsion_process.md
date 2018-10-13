
### markov property란,

현재의 상태로 미래상태를 예측할 수 있는 특성을 의미한다.

defined by (S,A,R,P,감마)

    S : state

    A : action

    R : reward

    P : state transition probability matrix

    감마 : discount factor 

### 알고리즘

  1. 첫 번째 스탭에서, step t=0, environment는 S 확률분포 P를 따라서 initial state를 지정한다.

  2. 첫 번째 state에서
  
    - agent는 action을 결정한다
    - enviroment는 reward를 구한다.
    - enviroment는 다음 state를 구한다.
    - agent는 reward와 다음 state를 받는다.

  3. 위의 과정을 max_episodies 만큼 반복한다.


### Q-learning

위의 알고리즘으로 Q-function을 구할 수 있지만, state-action pari의 reward 데이터가 필요하다. 즉, 확장가능성이 매우 떨어진다는 뜻이다.

예를 들어서, 슈퍼마리오 게임을 진행한다고 가정해보면 슈퍼마리오의 모든 위치에 대해서 모든 행동에 대한 reward가 필요하다는 것인데 이는 매우 비효율적이며, 현실적으로 불가능하다.

이런 점을 해결하기위해서 Q(s,a)를 함수로 근사하는데 이를 Q-learning이라고 부른다.

Q(s,a;theta)=Q(s,a)

theta는 함수의 parameter를 의미한다.

![Q-learning](https://wdc.objectstorage.softlayer.net/v1/AUTH_7046a6f4-79b7-4c6c-bdb7-6f68e920f6e5/Code-Articles/cc-reinforcement-learning-train-software-agent/images/fig03.png)


![Q-learning_loss](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRt6sqIGLjef-RnIa0H3wU2JDgPUtltlKtMGmioFFLh__pPoBD8)


1.
2.
3.
