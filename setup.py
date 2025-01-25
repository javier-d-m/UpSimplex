#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# setup.py
# Description: setup
# -----------------------------------------------------------------------------
#
# Started on  <Sat Jan 25,  14:39:00 2025 Javier Diaz Medina>
# Last update <Sat Jan 25,  14:39:00 2025 Javier Diaz Medina>
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
#     This file is part of UpSimplex
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
#
#     This code's initial source was made by Carlos Clavero Munoz, 2016
#     Here is the original repository and his information for future references.
#     Carlos Clavero Muñoz, c.clavero74@gmail.com, 'https://github.com/carlosclavero/PySimplex'
# -----------------------------------------------------------------------------
from distutils.core import setup
setup(
  name = 'UpSimplex',
  packages = ['UpSimplex'], 
  version = '1.0.0',
  description = 'This module contains tools to solve linear programming problems.',
  author = 'Javier Diaz Medina',
  author_email = 'javierediazem@gmail.com',
  url = 'https://github.com/javier-d-m/UpSimplex',
  keywords = ['simplex', 'linear', 'programming','rational','maths'], 
  classifiers = [],
)