[README.md](https://github.com/user-attachments/files/32830419/README.md)
# Detección de enfermedades en hojas de frijol con redes neuronales

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TU-USUARIO/deteccion-enfermedades-hojas-frijol/blob/main/notebooks/Proyecto_Bean_Leaf_Disease.ipynb)

Clasificación de imágenes de hojas de frijol en tres categorías (sana, mancha angular y roya) con **TensorFlow/Keras**. El proyecto compara perceptrones multicapa (MLP) y redes convolucionales (CNN), con y sin regularización por dropout, y explica las predicciones del mejor modelo con **SHAP**.

<img width="1000" height="633" alt="muestras_clases" src="https://github.com/user-attachments/assets/9bc7c724-2f17-458f-abcf-9240eb7d610c" />[README.md](https://github.com/user-attachments/files/32830501/README.md)


*Etiqueta 0 = `angular_leaf_spot` (mancha angular), 1 = `bean_rust` (roya), 2 = `healthy` (sana).*

## Objetivo

Diagnosticar de forma automática la enfermedad de una hoja de frijol a partir de una fotografía tomada en campo, y entender qué arquitecturas y qué hiperparámetros funcionan mejor con un conjunto de datos pequeño.

## Datos

| Aspecto | Detalle |
|---|---|
| Dataset | [Beans](https://www.tensorflow.org/datasets/catalog/beans) de TensorFlow Datasets (Makerere AI Lab) |
| Imágenes | 1,295 fotografías RGB de 500 × 500 píxeles |
| Clases | `angular_leaf_spot`, `bean_rust`, `healthy` (balanceadas) |
| Partición | Se reunieron las particiones originales y se dividieron en 80 / 10 / 10: **1,035 entrenamiento, 131 validación, 129 prueba** |

## Metodología

1. **Preprocesamiento:** normalización de píxeles a [0, 1] y reescalado a 128 × 128.
2. **Aumento de datos (solo entrenamiento):** volteo horizontal, rotaciones de 0°, 90°, 180° y 270°, y zoom aleatorio mediante recorte.
3. **Modelos comparados**, todos con Adam, entropía cruzada categórica, lotes de 64 y 40 épocas:
   - **MLP** de tres capas densas (512, 256 y 128 neuronas), unos 25.3 millones de parámetros.
   - **CNN** de cinco bloques convolución + max pooling y una capa densa de 64 neuronas, unos 110 mil parámetros.
   - Ambas con **dropout** (0.4, 0.5 y 0.6 en el MLP; 0.5 antes de la capa de salida en la CNN).
4. **Ajuste de la tasa de aprendizaje** (entre 5e-7 y 1e-3) en cada familia de modelos.
5. **Métricas:** accuracy, precisión macro y pérdida sobre el conjunto de prueba, además de curvas de entrenamiento y validación por época.
6. **Interpretabilidad:** valores SHAP sobre 15 imágenes de prueba con el mejor modelo.

## Resultados (conjunto de prueba, 129 imágenes)

| Modelo | Tasa de aprendizaje | Accuracy | Precisión macro |
|---|---|---|---|
| MLP | 1e-4 | 0.574 | 0.618 |
| MLP | 1e-3 | 0.519 | 0.495 |
| MLP | 5e-7 | 0.589 | 0.590 |
| CNN | 1e-4 | 0.667 | 0.743 |
| CNN | 1e-3 | 0.736 | 0.776 |
| CNN | 5e-4 | 0.736 | 0.774 |
| MLP + dropout 0.4 | 1e-4 | 0.465 | 0.427 |
| MLP + dropout 0.5 | 1e-4 | 0.333 | 0.111 |
| MLP + dropout 0.6 | 1e-4 | 0.333 | 0.111 |
| CNN + dropout 0.5 | 1e-4 | 0.643 | 0.711 |
| **CNN + dropout 0.5** | **1e-3** | **0.791** | **0.789** |
| CNN + dropout 0.5 | 5e-4 | 0.729 | 0.796 |

El mejor modelo es la **CNN con dropout de 0.5 y tasa de aprendizaje de 1e-3**, con accuracy de 0.791 y la menor pérdida de prueba (0.449).

<img width="855" height="558" alt="accuracy_por_epoch" src="https://github.com/user-attachments/assets/720a4e05-d664-4e1a-bb04-53d13a82017e" />


## Hallazgos

- **La CNN supera claramente al MLP:** el mejor MLP llega a 0.59 de accuracy y la mejor CNN a 0.79, con unas 230 veces menos parámetros. La convolución aprovecha la estructura espacial de la imagen, algo que un MLP pierde al aplanarla.
- **El dropout perjudica al MLP:** con probabilidades de 0.5 y 0.6 el accuracy cae a 0.333 y la precisión macro a 0.111, resultado consistente con un modelo que predice siempre la misma clase.
- **En la CNN el efecto del dropout es mixto:** mejora el mejor caso (de 0.736 a 0.791 con tasa de 1e-3), pero no ayuda con las otras dos tasas de aprendizaje.
- **Sin sobreajuste marcado:** en las curvas, el accuracy de entrenamiento y el de validación se mantienen cercanos y siguen subiendo hacia la época 40, lo que sugiere margen para entrenar más tiempo.

## Limitaciones

- El conjunto de prueba tiene solo 129 imágenes: cada una pesa cerca de 0.8 puntos porcentuales, así que diferencias como 0.79 frente a 0.74 equivalen a unas 7 imágenes.
- Los modelos se compararon y el mejor se eligió sobre el mismo conjunto de prueba; lo más riguroso es decidir con validación o validación cruzada y reservar la prueba para el modelo final.
- No se fijó una semilla aleatoria, por lo que los resultados pueden variar entre ejecuciones.

## Próximos pasos

- **Transferencia de aprendizaje** con una red preentrenada (ResNet50, EfficientNet, MobileNet), lo más probable para superar el 0.79 con tan pocos datos.
- Reportar métricas por clase y matrices de confusión, y evaluar con validación cruzada.
- Probar más aumentos de datos (brillo y contraste, que quedaron comentados en el código).
- Publicar una demo con Gradio o Streamlit para subir una foto y obtener el diagnóstico.

## Cómo reproducirlo

Recomendado en Google Colab con GPU (el proyecto se ejecutó con una T4) usando el botón de arriba.

Para correrlo localmente:

```bash
pip install tensorflow tensorflow-datasets scikit-learn numpy pandas matplotlib seaborn shap jupyter
jupyter notebook notebooks/Proyecto_Bean_Leaf_Disease.ipynb
```

Los datos se descargan automáticamente con `tfds.load('beans')`. La celda de SHAP es la más lenta: explicar las 15 imágenes toma más de 10 minutos.

## Créditos

Dataset Beans del Makerere AI Lab, distribuido a través de TensorFlow Datasets.

## Autor

José Pablo Fernández Agüero, Estadístico · [LinkedIn](https://www.linkedin.com/in/TU-PERFIL)
