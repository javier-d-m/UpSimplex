#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# IO.py
# Description: library with linear programming and maths services for Operation Research class
# -----------------------------------------------------------------------------
#
# Started on  <Sat Jan 28,  9:33:00 2025 Javier Diaz Medina>
# Last update <Sat Feb 11,  10:11:00 2025 Javier Diaz Medina>
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
from gilp import LP
from gilp import simplex_visual
warnings.filterwarnings('ignore')


############################ Rational operations ############################
def LP_Graphical_method():
    # Paso 1: Configuración del modelo
    # Preguntar al usuario el número de variables
    num_variables = int(input("Number of variables to create: "))
    if num_variables <= 0:
        raise ValueError("You should create at least one variable")
    elif num_variables > 3:
        raise ValueError("It is not posible to have more than 3 variables")
    variables = symbols(' '.join([f'x{i + 1}' for i in range(num_variables)]), real=True, nonnegative=True)
    SimboloVariable = [f"x{i+1}" for i in range(num_variables)]

    # Preguntar al usuario el número de restricciones
    # Preguntar por el número de restricciones
    num_restricciones = int(input("Number of constraints: "))
    if num_restricciones <= 0:
        raise ValueError("You should declare at least one constraint")

    # Preguntar si la función objetivo es maximizar o minimizar
    objetivo = input("What is the objective function? (max/min): ").strip().lower()
    if objetivo not in ["max", "min"]:
        raise ValueError("You should type 'max' to maximize or 'min' to minimize.")

    # Preguntar por los coeficientes de la función objetivo
    print("Please enter the OF coefficients for each variable:")
    c = Matrix([-float(input(f"Coefficient for {var}: ")) for var in variables])

    if objetivo == "max":
        c = -c  # Convertir a minimización

    # Inicializar matrices para las restricciones
    A = []
    b = []
    restricciones_usuario = []  # Guardar las restricciones como las declara el usuario

    # Preguntar por cada restricción
    for i in range(num_restricciones):
        print(f"Please enter the coefficients of Constraint {i + 1} following the order of variables:")
        restriccion = [float(input(f"Coefficient for {var}: ")) for var in variables]
        operador = input("Is <=, >= or =?: ").strip()
        if operador not in ["<=", ">=", "="]:
            raise ValueError("No valid operator. Use '<=', '>=', or '='.")
        rhs = float(input("Please type the RHS value for the constraint: "))
        # Guardar la restricción como la ingresó el usuario
        restricciones_usuario.append((restriccion, operador, rhs))

        #Ajustar la matriz A y el vector b según el operador
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
    print("\nEntered Model:")
    print("Objective Function: ", end="")
    if objetivo == "max":
        print("Max z =", " + ".join([f"{abs(c[i]):.3f}" + SimboloVariable[i] if abs(c[i]) % 1 != 0 else f"{int(abs(c[i]))}" + SimboloVariable[i] for i in range(len(c))]))
    else:
        print("Min z =", " + ".join([f"{abs(c[i]):.3f}" + SimboloVariable[i] if abs(c[i]) % 1 != 0 else f"{int(abs(c[i]))}" + SimboloVariable[i] for i in range(len(c))]))
    print("Subject to:")
    for i, (coefs, op, rhs) in enumerate(restricciones_usuario):
        restriccion_str = " + ".join([f"{coefs[j]:.3f}" +  SimboloVariable[j] if coefs[j] % 1 != 0 else f"{int(coefs[j])}" +  SimboloVariable[j] for j in range(len(coefs))])
        print(f"{restriccion_str} {op} {rhs}")

    A_list = np.array(A, dtype=np.float64)
    b_list = np.array(b, dtype=np.float64)
    c_list = np.array(c, dtype=np.float64)

    lp = LP (A = A_list, b = b_list,c = c_list)
    #solo acepta puro menor igual
    #Si es minimizar se debe cambiar la funcion objetivo a menos uno
    simplex_visual ( lp = lp, rule='dantzig' ).show()

def LP_Standard_Simplex():
    # Paso 1: Configuración del modelo
    # Preguntar al usuario el número de variables
    num_variables = int(input("Number of variables to create: "))
    if num_variables <= 0:
      raise ValueError("You should indicate at least one variable")
    variables = symbols(' '.join([f'x{i + 1}' for i in range(num_variables)]), real=True, nonnegative=True)
    SimboloVariable = [f"x{i+1}" for i in range(num_variables)]

    # Preguntar al usuario el número de restricciones
    # Preguntar por el número de restricciones
    num_restricciones = int(input("Number of constraints to create: "))
    if num_restricciones <= 0:
      raise ValueError("You should indicate at least one constraint")

    # Preguntar si la función objetivo es maximizar o minimizar
    objetivo = input("What is the objective function? (max/min): ").strip().lower()
    if objetivo not in ["max", "min"]:
        raise ValueError("You should type 'max' to maximize or 'min' to minimize.")

    # Preguntar por los coeficientes de la función objetivo
    print("Please enter the OF coefficients (for each variable) following the order of variables::")
    c = Matrix([-float(input(f"Coefficient for {var}: ")) for var in variables])

    if objetivo == "max":
        c = -c  # Convertir a minimización

    # Inicializar matrices para las restricciones
    A = []
    b = []
    restricciones_usuario = []  # Guardar las restricciones como las declara el usuario
    desigualdades = [] #guardar desigualdades declaradas por el usuario

    # Preguntar por cada restricción
    for i in range(num_restricciones):
        print(f"Please enter the coefficients of Constraint {i + 1} following the order of variables:")
        restriccion = [float(input(f"Coefficient for {var}: ")) for var in variables]
        operador = input("¿Is it <=, >= o =?: ").strip()
        if operador not in ["<=", ">=", "="]:
            raise ValueError("No valid operator. Use '<=', '>=', or '='.")
        rhs = float(input("Please enter the RHS value for the constraint: "))

        desigualdades.append(operador)

        # Guardar la restricción como la ingresó el usuario
        restricciones_usuario.append((restriccion, operador, rhs))

        A.append(restriccion)
        b.append(rhs)
        # Ajustar la matriz A y el vector b según el operador
        #if operador == "<=":
        #    A.append(restriccion)
        #    b.append(rhs)
        #elif operador == ">=":
        #    A.append([-coef for coef in restriccion])
        #    b.append(-rhs)
        #elif operador == "=":
        #    A.append(restriccion)
        #    b.append(rhs)
        #    A.append([-coef for coef in restriccion])
        #    b.append(-rhs)

    A = Matrix(A)
    b = Matrix(b)

    # Mostrar el modelo tal como fue ingresado
    print("\nEntered Model:")
    print("Objective Function: ", end="")
    if objetivo == "max":
        print("Max z =", " + ".join([f"{abs(c[i]):.3f}" + SimboloVariable[i] if abs(c[i]) % 1 != 0 else f"{int(abs(c[i]))}" + SimboloVariable[i] for i in range(len(c))]))
    else:
        print("Min z =", " + ".join([f"{abs(c[i]):.3f}" + SimboloVariable[i] if abs(c[i]) % 1 != 0 else f"{int(abs(c[i]))}" + SimboloVariable[i] for i in range(len(c))]))
    print("Subject to:")
    for i, (coefs, op, rhs) in enumerate(restricciones_usuario):
        restriccion_str = " + ".join([f"{coefs[j]:.3f}" +  SimboloVariable[j] if coefs[j] % 1 != 0 else f"{int(coefs[j])}" +  SimboloVariable[j] for j in range(len(coefs))])
        print(f"{restriccion_str} {op} {rhs}")


    # Paso 2: Transformar restricciones a forma estándar
    ContadorAuxiliar= 0
    ContadorArtificial = 0

    for i in range(A.rows):
        restriccion = " + ".join([f"{A[i, j]:.3f}" + SimboloVariable[j] if A[i, j] % 1 != 0 else f"{int(A[i, j])}" + SimboloVariable[j] for j in range(A.cols)])
        while True:
            tipo_variable = input(f"For the constraint {i + 1} ({restriccion}{desigualdades[i]}{b[i]:.3f}), what type of auxiliary variable you want to add? slack, surplus, artificial. Next constraint (n)").strip().lower()
            if tipo_variable == "slack" or tipo_variable == "h":
                ContadorAuxiliar += 1
                SimboloVariable.append(f's{ContadorAuxiliar}')
                nueva_columna = Matrix.zeros(A.rows, 1)
                nueva_columna[i] = 1.0
                A = A.row_join(nueva_columna)
                restriccion = " + ".join([f"{A[i, j]:.3f}" + SimboloVariable[j] if A[i, j] % 1 != 0 else f"{int(A[i, j])}" + SimboloVariable[j] for j in range(A.cols)])
                print(f"The slack variable s{ContadorAuxiliar} was added to constraint {i + 1}.")
                desigualdades[i] = '='
            elif tipo_variable == "surplus" or tipo_variable == "e":
                ContadorAuxiliar += 1
                SimboloVariable.append(f's{ContadorAuxiliar}')
                nueva_columna = Matrix.zeros(A.rows, 1)
                nueva_columna[i] = -1.0
                A = A.row_join(nueva_columna)
                restriccion = " + ".join([f"{A[i, j]:.3f}" + SimboloVariable[j] if A[i, j] % 1 != 0 else f"{int(A[i, j])}" + SimboloVariable[j] for j in range(A.cols)])
                print(f"The surplus variable s{i + 1} was added to constraint {i + 1}.")
                desigualdades[i] = '='
            elif tipo_variable == "artificial" or tipo_variable == "a":
                ContadorArtificial += 1
                SimboloVariable.append(f'a{ContadorArtificial}')
                nueva_columna = Matrix.zeros(A.rows, 1)
                nueva_columna[i] = 1.0
                A = A.row_join(nueva_columna)
                restriccion = " + ".join([f"{A[i, j]:.3f}" + SimboloVariable[j] if A[i, j] % 1 != 0 else f"{int(A[i, j])}" + SimboloVariable[j] for j in range(A.cols)])
                print(f"The artificial variable a{i + 1} was added to constraint {i + 1}.")
                desigualdades[i] = '='
            elif tipo_variable == "next" or tipo_variable == "n":
                break
            else:
                print("Incorrect. Please select between 'slack', 'surplus', 'artificial' o 'next'.")

    # Actualizar el vector c para incluir las variables adicionales
    c = c.col_join(Matrix.zeros(len(SimboloVariable)-num_variables, 1,dtype=float))

    # Mostrar el modelo en forma estándar
    print("\nModel in standard form:")
    if objetivo == "max":
        print("Max z - (", " + ".join([f"{abs(c[i]):.3f}" + SimboloVariable[i] if abs(c[i]) % 1 != 0 else f"{int(abs(c[i]))}" + SimboloVariable[i] for i in range(len(c)) if c[i] != 0]).replace("+ -", "- "), ") = 0")
    else:
        print("Min z - (", " + ".join([f"{abs(c[i]):.3f}" + SimboloVariable[i] if abs(c[i]) % 1 != 0 else f"{int(abs(c[i]))}" + SimboloVariable[i] for i in range(len(c)) if c[i] != 0]).replace("+ -", "- "), ") = 0")

    print("Subject to:")
    for i in range(A.rows):
        restriccion = " + ".join([f"{A[i, j]:.3f}" + SimboloVariable[j] if A[i, j] % 1 != 0 else f"{int(A[i, j])}" + SimboloVariable[j] for j in range(A.cols)])
        restriccion = restriccion.replace("+ -", "- ")  # Ajustar signos
        print(f"{restriccion} = {b[i]:.3f}" if b[i] % 1 != 0 else f"{restriccion} = {int(b[i])}" )

    FuncionObjetivo = np.array(-c if objetivo == "max" else c).flatten()
    Matriz = np.array(A)
    Recursos = np.array(b)

    if "a1" in SimboloVariable:
      UserAnswer = ""
      while UserAnswer != "Y" and UserAnswer != "N":
        UserAnswer = input("Do you want to use the Two Phase Method? (Y/N) ")
        if UserAnswer != "Y" and UserAnswer != "N":
          print("Please select a valid answer...")
      if UserAnswer == "Y":
        print("Starting Two Phase Method...")
        DosFasesSimplex(Matriz,FuncionObjetivo,Recursos,SimboloVariable)
      else:
        print("Finished activity")
    else:
      Simplex(Matriz,FuncionObjetivo,Recursos,SimboloVariable)

def Simplex(Matriz, FuncionObjetivo, Recursos, SimboloVariable):
    Matriz = np.vstack([FuncionObjetivo, Matriz]) #añadimos la funcion objetivo a la matriz de restricciones
    df = pd.DataFrame(Matriz, columns=SimboloVariable) #creamos un data frame con la matriz de restricciones

    truncate = np.vectorize(lambda x: float(f"{x:.3f}")) #Convertimos el dataframe en decimales
    df = df.apply(truncate)
    ######################### Caculamos las variables que forman la matriz identidad
    result_columns_with_index = []
    for col in df.columns:
        column_data = df[col].iloc[1:]  # Check for exactly one 1 in the column (ignoring the first row)
        ones_count = (column_data == 1.000).sum()
        if ones_count == 1:  # Ensure there's exactly one 1
            idx = column_data[column_data == 1.000].index[0] # Get the index of the row with the 1
            # Ensure the index of the 1 corresponds to the position in the DataFrame
            if (column_data == 0.000).sum() == len(df) - 2:  # All other values are 0
                result_columns_with_index.append((col, idx))

    row_index = [col for col, _ in sorted(result_columns_with_index, key=lambda x: x[1])] #assign result_columns_with_index but order it by row
    row_index.insert(0, 'z') #agregamos w al row=index
    Recursos = np.insert(Recursos,0, 0) #agregamos un cero simbolizando el valor de la funcion objetivo
    df["b"] = Recursos #insertamos la columna de 1recursos al data frame
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
    df.at['z', 'b'] = product_sum if product_sum > 0 else 0 # Assign the sum product result to the 'b' column for row Z, si es menor a cero se queda como cero

    ##########################         Imprimir data frame
    print("\n Simplex Method:")
    truncate = np.vectorize(lambda x: float(f"{x:.3f}"))
    df = df.apply(truncate)
    print(df.to_string(index=True)) #, float_format="%.3f"

    ######################## gauss jordan logic
    UserAnswer = False
    while UserAnswer==False:
        if input("Is the solution of the current tableau optimal? Y/N ") == "Y":
            UserAnswer = True
            break
        else:
            UserAnswer = False
        VariableEntrada = input("Plese select the Entering Variable (EV): ")
        VariableSalida = input("Please select the Leaving Variable (LV): ")
        FilaPivote = float(input("Please enter the factor to divide the pivot row: "))
        NumberOfRows = df.shape[0]
        df.loc[VariableSalida] = df.loc[VariableSalida]/FilaPivote #dividimos fila pivote por el número indicado
        truncate = np.vectorize(lambda x: float(f"{x:.3f}"))
        df = df.apply(truncate)
        print(df.to_string(index=True))
        MultipleValues = [
        float(input(f"Please enter the multiplier for row {VariableSalida} that will be SUBTRACTED from row {row_index[rows]}: "))
        if row_index[rows] != VariableSalida else None
        for rows in range(NumberOfRows)
        ]
        for i in range(NumberOfRows):
            if row_index[i] != VariableSalida:
                df.loc[row_index[i]] = df.loc[row_index[i]] - (df.loc[VariableSalida]*MultipleValues[i])

        #df = df.rename(index={VariableSalida: VariableEntrada})#Change index row name for the entry
        row_index = [VariableEntrada if x == VariableSalida else x for x in row_index]
        df.index = row_index
        truncate = np.vectorize(lambda x: float(f"{x:.3f}"))
        df = df.apply(truncate)
        print(df.to_string(index=True))
    print("Finished")

def DosFasesSimplex(Matriz, FuncionObjetivo, Recursos, SimboloVariable):
    #Nueva funcion objetivo W
    FuncionObjetivoAuxiliar = FuncionObjetivo.copy()
    for i in range(len(FuncionObjetivoAuxiliar)):
        if SimboloVariable[i].find("a") == -1:
            FuncionObjetivoAuxiliar[i] = 0
        else:
            FuncionObjetivoAuxiliar[i] = -1

    Matriz = np.vstack([FuncionObjetivoAuxiliar, Matriz]) #añadimos la funcion objetivo a la matriz de restricciones
    df = pd.DataFrame(Matriz, columns=SimboloVariable) #creamos un data frame con la matriz de restricciones

    truncate = np.vectorize(lambda x: float(f"{x:.3f}")) #Convertimos el dataframe en decimales
    df = df.apply(truncate)

    ######################### Acomodamos las columnas para dejar al final a las variables auxiliares
    ColumnasArtificiales = [col for col in df.columns if col.find('a')] # Identificar columnas que terminan en 'a' y las que no
    ColumnasNoArtificiales = [col for col in df.columns if not col.find('a')] # Reordenar las columnas: primero las que no terminan en 'a', luego las que sí
    OrdenColumnas = ColumnasArtificiales + ColumnasNoArtificiales # Reorganizar el DataFra me
    df = df[OrdenColumnas]
    SimboloVariable = OrdenColumnas # Actualizar la lista de símbolos de variable

    ######################### Calculamos cuales variables forman la matriz identidad
    result_columns_with_index = []
    for col in df.columns:
        column_data = df[col].iloc[1:]  # Check for exactly one 1 in the column (ignoring the first row)
        ones_count = (column_data == 1.000).sum()
        if ones_count == 1:  # Ensure there's exactly one 1
            idx = column_data[column_data == 1.000].index[0] # Get the index of the row with the 1
            # Ensure the index of the 1 corresponds to the position in the DataFrame
            if (column_data == 0.000).sum() == len(df) - 2:  # All other values are 0
                result_columns_with_index.append((col, idx))

    row_index = [col for col, _ in sorted(result_columns_with_index, key=lambda x: x[1])] #assign result_columns_with_index but order it by row
    row_index.insert(0, 'w') #agregamos w al row=index
    Recursos = np.insert(Recursos,0, 0) #agregamos un cero simbolizando el valor de la funcion objetivo
    df["b"] = Recursos #insertamos la columna de 1recursos al data frame

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
    df.at['w', 'b'] = product_sum if product_sum > 0 else 0 # Assign the sum product result to the 'b' column for row Z, si es menor a cero se queda como cero
    ##########################         Imprimir data frame
    print("\n Two Phase Simplex Method:")
    truncate = np.vectorize(lambda x: float(f"{x:.3f}"))
    df = df.apply(truncate)
    print(df.to_string(index=True))

    ######################## remove initial -1 in artificial variables
    MultipleValues = [
    float(input(f"Please type the multiplier for row {rows} that will be substracted from row W: "))
    for rows in row_index
    if rows.find("a") == False
    ]
    for i in range(len(MultipleValues)):
        df.loc['w'] = df.loc['w'] - (df.loc['a'+str(i+1)]*MultipleValues[i])
    truncate = np.vectorize(lambda x: float(f"{x:.3f}"))
    df = df.apply(truncate)
    print(df.to_string(index=True))

    ######################## gauss jordan logic
    UserAnswer = False
    while UserAnswer==False:
        if input("Is the current tableau ready to move to Phase 2?") == "Y":
            UserAnswer = True
            break
        else:
            UserAnswer = False
        VariableEntrada = input("Please type the Entering Variable (EV): ")
        VariableSalida = input("Please select the Leaving Variable (LV): ")
        FilaPivote = float(input("Please type the multiplier to divide the Pivot row: "))
        NumberOfRows = df.shape[0]
        df.loc[VariableSalida] = df.loc[VariableSalida]/FilaPivote #dividimos fila pivote por el número indicado
        truncate = np.vectorize(lambda x: float(f"{x:.3f}"))
        df = df.apply(truncate)
        print(df.to_string(index=True))
        MultipleValues = [
        float(input(f"Please type the multiplier to for row {VariableSalida} that will be SUBTRACTED from row {row_index[rows]}: "))
        if row_index[rows] != VariableSalida else None
        for rows in range(NumberOfRows)
        ]
        for i in range(NumberOfRows):
            if row_index[i] != VariableSalida:
                df.loc[row_index[i]] = df.loc[row_index[i]] - (df.loc[VariableSalida]*MultipleValues[i])

        row_index = [VariableEntrada if x == VariableSalida else x for x in row_index] #Change index row name for the entry
        df.index = row_index
        truncate = np.vectorize(lambda x: float(f"{x:.3f}"))
        df = df.apply(truncate)
        print(df.to_string(index=True))

    print("End of the Phase 1...")
    CantidadVariablesArtificiales = 0
    for letter in SimboloVariable:
        if 'a' in letter:
            CantidadVariablesArtificiales += 1

    Matriz = df.iloc[1:, :-(CantidadVariablesArtificiales+1)]
    FuncionObjetivo = FuncionObjetivo[:-CantidadVariablesArtificiales]
    Recursos = df.iloc[1:]['b']
    SimboloVariable = SimboloVariable[:-CantidadVariablesArtificiales]

    Simplex(Matriz,FuncionObjetivo,Recursos,SimboloVariable)

