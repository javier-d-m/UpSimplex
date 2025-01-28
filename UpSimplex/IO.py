#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# IO.py
# Description: library with linear programming and maths services for Operation Research class
# -----------------------------------------------------------------------------
#
# Started on  <Sat Jan 28,  9:33:00 2025 Javier Diaz Medina>
# Last update <Sat Jan 28,  9:33:00 2025 Javier Diaz Medina>
# -----------------------------------------------------------------------------
#
# $Id:: $
# $Date:: $
# $Revision:: $
# -----------------------------------------------------------------------------
#
# Made by Javier Diaz Medina
# 
#

# -----------------------------------------------------------------------------
#      This file is part of UpSimplex
#
#     UpSimplex is free software: you can redistribute it and/or modify it under
#     the terms of the GNU General Public License as published by the Free
#     Software Foundation, either version 3 of the License, or (at your option)
#     any later version.
#
#     UpSimplex is distributed in the hope that it will be useful, but WITHOUT ANY
#     WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
#     FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
#     details.
#
#     You should have received a copy of the GNU General Public License along
#     with UpSimplex.  If not, see <http://www.gnu.org/licenses/>.
#       
#     Copyright Javier Diaz Medina, 2025
# -----------------------------------------------------------------------------

import sys
import os.path
import io as io
import types
from sympy import symbols, Matrix
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


############################ Rational operations ############################


def PL_FormaEstandar():
    # Paso 1: Configuración del modelo
    # Preguntar al usuario el número de variables
    num_variables = int(input("¿Cuántas variables deseas utilizar? "))
    if num_variables <= 0:
      raise ValueError("Debes indicar al menos una variable")
    variables = symbols(' '.join([f'x{i + 1}' for i in range(num_variables)]), real=True, nonnegative=True)
    SimboloVariable = [f"x{i+1}" for i in range(num_variables)]

    # Preguntar al usuario el número de restricciones
    # Preguntar por el número de restricciones
    num_restricciones = int(input("¿Cuántas restricciones deseas crear? "))
    if num_restricciones <= 0:
      raise ValueError("Debes indicar al menos una restricción")

    # Preguntar si la función objetivo es maximizar o minimizar
    objetivo = input("¿Deseas maximizar o minimizar la función objetivo? (max/min): ").strip().lower()
    if objetivo not in ["max", "min"]:
        raise ValueError("Debes especificar 'max' para maximizar o 'min' para minimizar.")

    # Preguntar por los coeficientes de la función objetivo
    print("Ingresa los coeficientes de la función objetivo en el orden de las variables:")
    c = Matrix([-float(input(f"Coeficiente de {var}: ")) for var in variables])

    if objetivo == "max":
        c = -c  # Convertir a minimización

    # Inicializar matrices para las restricciones
    A = []
    b = []
    restricciones_usuario = []  # Guardar las restricciones como las declara el usuario

    # Preguntar por cada restricción
    for i in range(num_restricciones):
        print(f"Ingresa los coeficientes de la restricción {i + 1} en el orden de las variables:")
        restriccion = [float(input(f"Coeficiente de {var}: ")) for var in variables]
        operador = input("¿Es <=, >= o =?: ").strip()
        if operador not in ["<=", ">=", "="]:
            raise ValueError("Operador no válido. Usa '<=', '>=', o '='.")
        rhs = float(input("Ingresa el lado derecho de la restricción: "))

        # Guardar la restricción como la ingresó el usuario
        restricciones_usuario.append((restriccion, operador, rhs))

        # Ajustar la matriz A y el vector b según el operador
        if operador == "<=":
            A.append(restriccion)
            b.append(rhs)
        elif operador == ">=":
            A.append([-coef for coef in restriccion])
            b.append(-rhs)
        elif operador == "=":
            A.append(restriccion)
            b.append(rhs)
            A.append([-coef for coef in restriccion])
            b.append(-rhs)

    A = Matrix(A)
    b = Matrix(b)

    # Mostrar el modelo tal como fue ingresado
    print("\nModelo ingresado:")
    print("Función objetivo: ", end="")
    if objetivo == "max":
        print("Max z =", " + ".join([f"{abs(c[i]):.3f}" if abs(c[i]) % 1 != 0 else f"{int(abs(c[i]))}" + SimboloVariable[i] for i in range(len(c))]))
    else:
        print("Min z =", " + ".join([f"{abs(c[i]):.3f}" if abs(c[i]) % 1 != 0 else f"{int(abs(c[i]))}" + SimboloVariable[i] for i in range(len(c))]))
    print("Sujeto a:")
    for i, (coefs, op, rhs) in enumerate(restricciones_usuario):
        restriccion_str = " + ".join([f"{coefs[j]:.3f}" if coefs[j] % 1 != 0 else f"{int(coefs[j])}" +  SimboloVariable[j] for j in range(len(coefs))])
        print(f"{restriccion_str} {op} {rhs}")


    # Paso 2: Transformar restricciones a forma estándar
    ContadorHolgura = 0
    ContadorExceso = 0
    ContadorArtificial = 0
    for i in range(A.rows):
        restriccion = " + ".join([f"{A[i, j]:.3f}" if A[i, j] % 1 != 0 else f"{int(A[i, j])}" + SimboloVariable[j] for j in range(A.cols)])
        while True:
            tipo_variable = input(f"Para la restricción {i + 1} ({restriccion} <= {b[i]:.3f}), ¿qué tipo de variable deseas agregar (holgura, exceso, artificial)? ").strip().lower()
            if tipo_variable == "holgura":
                ContadorHolgura += 1
                SimboloVariable.append(f's{ContadorHolgura}')
                nueva_columna = Matrix.zeros(A.rows, 1)
                nueva_columna[i] = 1.0
                A = A.row_join(nueva_columna)
                print(f"Se agregó la variable de holgura s{ContadorHolgura} a la restricción {i + 1}.")
                break
            elif tipo_variable == "exceso":
                ContadorExceso += 1
                SimboloVariable.append(f'e{ContadorExceso}')
                nueva_columna = Matrix.zeros(A.rows, 1)
                nueva_columna[i] = -1.0
                A = A.row_join(nueva_columna)
                print(f"Se agregó la variable de exceso e{i + 1} a la restricción {i + 1}.")
                break
            elif tipo_variable == "artificial":
                ContadorArtificial += 1
                SimboloVariable.append(f'a{ContadorArtificial}')
                nueva_columna = Matrix.zeros(A.rows, 1)
                nueva_columna[i] = 1.0
                A = A.row_join(nueva_columna)
                print(f"Se agregó la variable artificial a{i + 1} a la restricción {i + 1}.")
                break
            else:
                print("Entrada incorrecta. Por favor, selecciona entre 'holgura', 'exceso' o 'artificial'.")

    # Actualizar el vector c para incluir las variables adicionales
    c = c.col_join(Matrix.zeros(len(SimboloVariable)-num_variables, 1,dtype=float))

    # Mostrar el modelo en forma estándar
    print("\nModelo en forma estándar:")
    if objetivo == "max":
        print("Max z - (", " + ".join([f"{abs(c[i]):.3f}" if abs(c[i]) % 1 != 0 else f"{int(abs(c[i]))}" + SimboloVariable[i] for i in range(len(c)) if c[i] != 0]).replace("+ -", "- "), ") = 0")
    else:
        print("Min z - (", " + ".join([f"{abs(c[i]):.3f}" if abs(c[i]) % 1 != 0 else f"{int(abs(c[i]))}" + SimboloVariable[i] for i in range(len(c)) if c[i] != 0]).replace("+ -", "- "), ") = 0")

    print("Sujeto a:")
    for i in range(A.rows):
        restriccion = " + ".join([f"{A[i, j]:.3f}" if A[i, j] % 1 != 0 else f"{int(A[i, j])}" + SimboloVariable[j] for j in range(A.cols)])
        restriccion = restriccion.replace("+ -", "- ")  # Ajustar signos
        print(f"{restriccion} = {b[i]:.3f}" if b[i] % 1 != 0 else f"{restriccion} = {int(b[i])}" )

    FuncionObjetivo = np.array(-c if objetivo == "max" else c).flatten()
    Matriz = np.array(A)
    Recursos = np.array(b)
    Simplex(Matriz,FuncionObjetivo,Recursos,SimboloVariable)

def Simplex(Matriz, FuncionObjetivo, Recursos, SimboloVariable):
  Matriz = np.vstack([FuncionObjetivo, Matriz]) #añadimos la funcion objetivo a la matriz de restricciones
  df = pd.DataFrame(Matriz, columns=SimboloVariable) #creamos un data frame con la matriz de restricciones
  row_index = df.columns[df.iloc[0] == 0].tolist() #extract the name of variables with zero to add them later as index
  row_index.insert(0, 'z')
  Recursos = np.insert(Recursos,0, 0) #agregamos un cero simbolizando el valor de la funcion objetivo
  df["b"] = Recursos #insertamos la columna de recursos al data frame
  df.index = row_index #cambiamos el indice por los valores de las variables
  df = df.astype(float)

  #########################         Calculate Z value
  # Calculate the sum product for row Z (ignoring row Z itself)
  first_row_values = df.iloc[0, :-1]  # Get the first row (ignoring 'b' column)
  last_column_values = df['b']  # Get the last column (b column)
  # Initialize a variable to store the sum product result
  product_sum = 0
  # Iterate through the rows and calculate the sum product
  for col in df.columns[:-1]:  # Exclude 'b' column for multiplication
      for idx, value in df.iterrows():
          if col == idx:  # If column name matches row index
              product_sum += first_row_values[col] * last_column_values[idx]
  # Assign the sum product result to the 'b' column for row Z
  df.at['z', 'b'] = product_sum
  ##########################         Calculate Z value
  print("\n Método Simplex:")
  print(df.to_string(index=True, float_format="%.3f"))

  ######################## gauss jordan logic
  UserAnswer = False
  while UserAnswer==False:
    if input("¿Esta solución es la optima? Escribe Si o No ") == "Si":
      UserAnswer = True
      break
    else:
      UserAnswer = False
    VariableEntrada = input("Ingresa la variable de Entrada: ")
    VariableSalida = input("Ingresa la variable de Salida: ")
    FilaPivote = float(input("Ingresa el valor por el se dividirá la fila pivote: "))
    NumberOfRows = df.shape[0]
    df.loc[VariableSalida] = df.loc[VariableSalida]/FilaPivote #dividimos fila pivote por el número indicado
    print(df.to_string(index=True, float_format="%.3f"))
    MultipleValues = [
    float(input(f"Escribe el multiplicador para la fila {VariableSalida} que se restará a la fila {row_index[rows]}: "))
    if row_index[rows] != VariableSalida else None
    for rows in range(NumberOfRows)
    ]
    for i in range(NumberOfRows):
      if row_index[i] != VariableSalida:
        df.loc[row_index[i]] = df.loc[row_index[i]] - (df.loc[VariableSalida]*MultipleValues[i])

    #df = df.rename(index={VariableSalida: VariableEntrada})#Change index row name for the entry
    row_index = [VariableEntrada if x == VariableSalida else x for x in row_index]
    df.index = row_index
    print(df.to_string(index=True, float_format="%.3f"))
  print("Proceso finalizado")

