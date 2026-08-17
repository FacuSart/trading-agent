# Copiloto experimental de trading

Un MVP local para investigar swing trading diario: descarga precios, crea indicadores, entrena modelos cronológicos para 5, 10 y 20 ruedas y veta señales que no superan controles mínimos.

> **No es asesoramiento financiero ni está listo para dinero real.** La primera meta es comprobar que el proceso es honesto y repetible en simulación.

## Qué hace hoy

- Usa únicamente velas diarias completas de Yahoo Finance y descarta conservadoramente la vela fechada hoy.
- Calcula momentum, medias exponenciales, RSI, ATR, volatilidad y volumen relativo.
- Entrena regresiones logísticas regularizadas para horizontes de 5, 10 y 20 ruedas.
- Usa una ventana móvil máxima de cinco años y reentrena al inicio de cada semana.
- Purga cualquier etiqueta cuyo resultado todavía no hubiera finalizado en la fecha predicha.
- Decide al cierre y simula la ejecución en la apertura siguiente.
- Cobra el costo al entrar y al salir, limita cada posición a 20 ruedas y permite limitar el capital expuesto.
- Evalúa los tres modelos sobre las mismas últimas 252 ruedas.
- Muestra AUC, Brier, drawdown, Sharpe y resultados por operaciones cerradas.
- Bloquea `COMPRAR` si menos de dos horizontes superan validación o si no existe consenso.
- Permite elegir activos comunes desde una lista o escribir cualquier símbolo de Yahoo Finance.
- Tiene en cuenta si actualmente estás comprado para diferenciar comprar, mantener y salir.

## Puesta en marcha (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m streamlit run app.py
```

Se abrirá el navegador. Para comenzar se recomienda `SPY` o `QQQ`, costo estimado de `0.10%`, asignación máxima de `25%` y únicamente capital simulado. El sistema descarga diez años para evaluar, pero cada ajuste utiliza como máximo los cinco años anteriores.

## Ejecutar pruebas

```powershell
.venv\Scripts\python -m pytest -q
```

## Cómo interpretar la señal

- **COMPRAR:** al menos dos horizontes validados coinciden en una entrada y la dispersión es aceptable.
- **MANTENER POSICIÓN:** el sistema ya estaba dentro y no apareció una salida.
- **SALIR:** aparece consenso defensivo o se alcanzan 20 ruedas de duración.
- **ESPERAR:** no existe una ventaja suficiente según el modelo.
- **SEÑAL INCIERTA:** los horizontes discrepan; no abrir una operación nueva.
- **MODELO NO VALIDADO:** no supera los controles de AUC, Brier, retorno y cantidad mínima de operaciones; no abrir una operación nueva.

El puntaje alcista no es una probabilidad garantizada. La aplicación lo llama *puntaje* y exige que su Brier mejore frente a una predicción base. El modelo no “sabe” el futuro y las relaciones históricas pueden desaparecer.

Presionar **Analizar ahora** repetidamente no acumula aprendizaje. Las señales cambian con una vela nueva o con parámetros distintos. Los umbrales están dentro de **Configuración avanzada** para desalentar su ajuste mirando el backtest.

## Criterios automáticos de validación

Cada horizonte necesita, sobre datos walk-forward:

- al menos 100 observaciones evaluables;
- ROC AUC mínimo de 0.52;
- Brier menor que el predictor base cronológico;
- retorno simulado positivo después de costos;
- al menos tres operaciones cerradas.

Son filtros experimentales, no una prueba de rentabilidad futura. El backtest de referencia usa la misma asignación de capital que la estrategia para que la comparación de riesgo sea coherente.

## Camino hasta la automatización

1. **MVP de investigación (actual):** datos históricos, señal explicable y backtest cronológico.
2. **Validación:** fijar mercado y horizonte, separar un período final intocable, comparar con reglas simples y ejecutar paper trading durante varias semanas/meses.
3. **Operación asistida:** proceso programado, base de datos y alertas por Telegram/email; una persona confirma cada orden.
4. **Paper trading automatizado:** conectar un broker en sandbox, con tamaño máximo, stop diario, registros y botón de apagado.
5. **Dinero real:** solo tras criterios de aceptación definidos de antemano; empezar con capital pequeño y mantener límites fuera del modelo.

Para el siguiente paso habrá que elegir mercado (cripto, acciones o forex), broker disponible en tu país, temporalidad y capital/riesgo máximo. Esas decisiones cambian los datos, costos y ejecución.
