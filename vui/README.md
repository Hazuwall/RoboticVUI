## VUI

### Сборка

`python setup.py sdist`

### Установка

1. `pip install -e vui`

2. Установить [graphviz](https://graphviz.gitlab.io/download/)

   Добавить путь к папке bin в PATH

3. Установить `pyaudio`

   - Для Python >=3.7:

     `pip install pipwin`

     `pipwin install pyaudio`

   - Для Python <3.7:

     `pip install pyaudio`

### Ошибки:

1. Cannot convert a symbolic Tensor ({}) to a numpy

Изменить участок кода

```
def _constant_if_small(value, shape, dtype, name):
    try:
        if np.prod(shape) < 1000:
            return constant(value, shape=shape, dtype=dtype, name=name)
    except TypeError:
        # Happens when shape is a Tensor, list with Tensor elements, etc.
        pass
    except NotImplementedError:
        pass
    return None
```
