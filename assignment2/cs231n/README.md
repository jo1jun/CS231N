아래 계산 그래프들은 node들을 합쳐 더 간단하게 표현하고 더 효율적으로 계산할 수 있다. (+batch 처리)
계산그래프 전공은 아니기에 더 단순화, 일반화 시키진 않았다. 추후에 필요하다면 수정할 것. 
처음 학습 과정에서 명확한 이해를 위해 그려볼 순 있지만 이미 잘 구현된 framwork 들을 활용하면 된다.

## convolution Computational Graph (non-batch, non-filter wise)
![conv](https://user-images.githubusercontent.com/68524289/112561520-c3301680-8e18-11eb-8be7-478f4084e71b.PNG)
## max_pooling Computational Graph (non-batch, non-channel wise)
![pool](https://user-images.githubusercontent.com/68524289/112561559-d6db7d00-8e18-11eb-83f2-e6f6f7d47ad7.PNG)