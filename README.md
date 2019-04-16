## Overview

�˲ֿ�����Ҷ� Udacity �� Capstone ��Ŀ�� [**Deep Knowledge Tracing**](https://github.com/chrispiech/DeepKnowledgeTracing) ʵ�֡�

## Ŀ��

������ѵ��LSTM���磬��Ԥ��ѧ����ȷ�ش�����δ����������ĸ��ʡ�ʹ�õ��� [**ASSISTments Skill-builder data 2009-2010**](https://sites.google.com/site/assistmentsdata/home/assistment-2009-2010-data/skill-builder-data-2009-2010) �������ݼ���

## ���

����ͨ�������޸��������ã���ȡ�������֤��ʧ��

| Test Data (%) | AUC |
| --- | --- |
| 20% | 0,85 |

�����ڡ�Log���ļ������ҵ�ÿ�γ��ԵĽ�������ú�ģ��Ȩ�ء�

## Ҫ��

����ҪPython 3.x x64����������Щ��Ŀ��

�������û�а�װPython����������װPython �� [Anaconda](https://www.anaconda.com/download/) ���а棬������������Щ��Ŀ������������������

��Ҳ���Դ�[����](https://www.python.org/downloads/)��װPython 3.x x64

## ˵��

1. ��¡�洢�Ⲣ���������ص��ļ��С�
```	
git clone https://github.com/lccasagrande/Deep-Knowledge-Tracing.git
cd Deep-Knowledge-Tracing
```

2. ��װ����İ���
	- ����Ѱ�װTensorFlow������룺
	```
	pip install -e .
	```
	- ���Ҫʹ��TensorFlow-GPU���а�װ���밴��[��ָ��](https://www.tensorflow.org/install/) ���ϵͳ�ϱ����NVIDIA�����Ȼ����룺
	```
	pip install -e .[tf_gpu]
	```
	- ���Ҫʹ��Tensorflow-CPU���а�װ������룺
	```
	pip install -e .[tf]
	```

3. ������src�ļ��в��� notebook.
```	
cd src
jupyter notebook DKT.ipynb
```

