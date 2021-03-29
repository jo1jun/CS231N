아래 계산그래프를 기반으로 위의 code들을 구현하였다.

아래 계산 그래프들은 node들을 합쳐 더 간단하게 표현하고 더 효율적으로 계산할 수 있다. (+batch 처리)

계산그래프 전공은 아니기에 더 단순화, 일반화 시키진 않았다. 

추후에 필요하다면 수정할 것.

처음 학습 과정에서 명확한 이해를 위해 그려볼 순 있지만 이미 잘 구현된 framwork 들을 활용하면 된다.

## convolution Computational Graph (non-batch, non-filter wise)
![conv](https://user-images.githubusercontent.com/68524289/112561520-c3301680-8e18-11eb-8be7-478f4084e71b.PNG)
## max_pooling Computational Graph (non-batch, non-channel wise)
![pool](https://user-images.githubusercontent.com/68524289/112561559-d6db7d00-8e18-11eb-83f2-e6f6f7d47ad7.PNG)
## batch_norm (batch)
![batch_norm1](https://user-images.githubusercontent.com/68524289/112711936-7da24500-8f0f-11eb-8985-1cad989b983e.png)
## batch_norm_alt (batch)
![batch_norm2](https://user-images.githubusercontent.com/68524289/112711955-a3c7e500-8f0f-11eb-9643-285abc9c8623.png)
## layer_norm (batch)
![layernorm](https://user-images.githubusercontent.com/68524289/112798854-77939c00-90a8-11eb-9337-7cb122e5d87c.png)
## spatial_batch_norm (batch)
![spatial_batchnorm](https://user-images.githubusercontent.com/68524289/112798865-7bbfb980-90a8-11eb-94fc-bc7d4edd08ba.png)
## sptatial_group_norm (batch)
![spatial_groupnorm](https://user-images.githubusercontent.com/68524289/112798881-81b59a80-90a8-11eb-9166-9f1bbe2fb06f.png)
