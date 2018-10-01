# Preparing TF protobuf(pb) for Android

## Prerequisites
* Bazel
* Tensorflow

## Installing Bazel on Ubuntu
* bazel을 이용한 build는 Linux(혹은 MacOS)환경에서 하는 것을 추천
* 패키지 설치 (zlib1g에서 5번째 글자는 숫자 1이다.)
    ```bash
    sudo apt-get install pkg-config zip g++ zlib1g-dev unzip
    ```

* bazel-&lt;version&gt;-installer-linux-x86_64.sh를 [Bazel releasese page on Github](https://github.com/bazelbuild/bazel/releases)에서 다운로드
  
* bazel installer 실행
    ```bash
    chmod +x bazel-<version>-installer-linux-x86_64.sh
    ./bazel-<version>-installer-linux-x86_64.sh --user
    ```

* 위처럼 --user 플래그로 실행하면 Bazel executable이 $HOME/bin에 설치된다. PATH 설정을 해준다.  
  AWS ubuntu의 경우 HOME은 home/ubuntu이다.
    ```bash
    export PATH="$PATH:$HOME/bin"
    ```

* command-line interpreter에서 tab completion을 할 수 있도록 complete.bash를 실행한다.
    ```bash
    source /home/ubuntu/.bazel/bin/bazel-complete.bash
    ```

## Writing Graph
* 코드 내에서 write_graph를 이용해 .pb 혹은 .pbtxt 파일 만들기.
* pbtxt파일을 만드는 경우에는 as_text=True로 설정해준다.
    ```bash
    tf.train.write_graph(sess.graph_def, log_dir='graph', name='full_graph.pb', as_text=False)
    ```
    
## 0. Git Clone
```bash
git clone https://github.com/tensorflow/tensorflow
cd tensorflow
```

## 1. Freezing
* freeze_graph.py를 이용해 variable을 constant로 만들기. pb(txt)파일과 ckpt파일(세 가지) 필요!
* input_checkpoint에는 체크포인트의 prefix만 적어준다.
    ```bash
    python tensorflow/python/tools/freeze_graph.py \
    --input_graph=full_graph.pb \
    --input_checkpoint=model.ckpt-1000 \
    --output_graph=frozen_graph.pb \
    --output_node_names=softmax
    ```
    
* 위 코드를 실행하면 cannot import name 'checkpoint_management'라는 에러가 뜬다.  
  그러면 freeze_graphy.py에 들어가서  
  (58줄) from tensorflow.python.training import checkpoint_management를 지우고  
  (127줄) not checkpoint_management.checkpoint_exists(input_checkpoint)): 를  
          not checkpoint_exists(input_checkpoint)): 로 바꿔준다.

* output node(와 이후 과정에서 input node)의 이름을 알기 위해서는 tensorboard를 사용한다. 
  아래의 Summarizing에서 확인할 수도 있다.
    ```bash
    tensorboard --logdir=./logs
    ```
    
## 2. Summarizing
* 입력과 출력 정보, 모델 크기를 살펴보기 위해 작업 전,후에는 수시로 graph summary를 찍어보는 것이 좋다.  
    ```bash
    bazel build tensorflow/tools/graph_transforms:summarize_graph
    
    bazel-bin/tensorflow/tools/graph_transforms/summarize_graph --in_graph=frozen_graph.pb
    ```

## 3. Graph Transforms Tool

### 3-1. Stripping
* strip_unused_nodes로 학습에만 필요한 노드 제거
    ```bash
    bazel build tensorflow/tools/graph_transforms:transform_graph
    
    bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
    --in_graph=frozen_graph.pb \
    --out_graph=optimized_frozen_graph.pb \
    --inputs='Mul' \
    --outputs='softmax' \
    --transforms=' \
      strip_unused_nodes(type=float, shape="1,299,299,3") \
      fold_constants(ignore_errors=true) \
      fold_batch_norms'
    ```
    
* 문서들을 보면 --transforms에서 fold_old_batch_norms도 사용하라고 나오지만,
  이는 예전 텐서플로우로 모델을 만든 경우에만 사용한다. 지금 버전에 사용하면 에러가 난다.

### 3-2. Quantizing
* quantize_weights로 float32를 8-bit int로 바꿔주기
    ```bash
    bazel build tensorflow/tools/graph_transforms:transform_graph
    
    bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
    --in_graph=optimized_frozen_graph.pb \
    --out_graph=quantized_frozen_graph.pb \
    --inputs='Mul' \
    --outputs='softmax' \
    --transforms='quantize_weights'
    ```
* round_weights를 사용하면 float32로 저장되지만, 단순한 round를 통해 더 높은 효율의 압축이 가능하다.

### 3-3. Etc.
* 추가적인 graph transforms tool들을 사용할 수 있다.  
  [Graph Transforms](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms) 참고.

## 4. Benchmark
* 속도 확인
    ```bash
    bazel build -c opt tensorflow/tools/benchmark:benchmark_model
    
    bazel-bin/tensorflow/tools/benchmark/benchmark_model \
    --graph=/tmp/quantized_frozen_graph.pb \
    --input_layer="Mul:0" \
    --input_layer_shape="1,299,299,3" \
    --input_layer_type="float" \
    --output_layer="softmax:0" \
    --show_run_order=false \
    --show_time=false \
    --show_memory=false \
    --show_summary=true \
    --show_flops=true \
    --logtostderr
    ```

## References
https://www.tensorflow.org/lite/tfmobile/prepare_models  
https://norman3.github.io/papers/docs/building_mobile_app_with_tf.html
