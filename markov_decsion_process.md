
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


