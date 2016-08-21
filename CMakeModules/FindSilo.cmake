# Copyright (c) 2008-2012 Sandia Corporation, Kitware Inc.
# Copyright (c) 2014-2014 Andreas Schäfer
#
# Sandia National Laboratories, New Mexico
# PO Box 5800
# Albuquerque, NM 87185
#
# Kitware Inc.
# 28 Corporate Drive
# Clifton Park, NY 12065
# USA
#
# Andreas Schäfer
# Informatik 3
# Martensstr. 3
# 91058 Erlangen
# Germany
#
# Under the terms of Contract DE-AC04-94AL85000, there is a
# non-exclusive license for use of this work by or on behalf of the
# U.S. Government.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the
#    distribution.
#
#  * Neither the name of Kitware nor the names of any contributors may
#    be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ========================================================================
#
# Try to find Silo library and headers. Define Silo_ROOT if Silo is
# installed in a non-standard directory.
#
# This file sets the following variables:
#
# Silo_INCLUDE_DIR, where to find silo.h, etc.
# Silo_LIBRARIES, the libraries to link against
# Silo_FOUND, If false, do not try to use Silo.
#
# Also defined, but not for general use are:
# Silo_LIBRARY, the full path to the silo library.
# Silo_INCLUDE_PATH, for CMake backward compatibility

FIND_PATH( Silo_INCLUDE_DIR silo.h
  PATHS /usr/local/include
  /usr/include
  ${Silo_ROOT}/include
)

FIND_LIBRARY( Silo_LIBRARY NAMES siloh5 silo
  PATHS /usr/lib
  /usr/lib64
  /usr/local/lib
  ${Silo_ROOT}/lib
  ${Silo_ROOT}/lib64
)

SET(Silo_FOUND "NO" )
IF(Silo_INCLUDE_DIR)
  IF(Silo_LIBRARY)

    SET(Silo_LIBRARIES ${Silo_LIBRARY})
    SET(Silo_FOUND "YES" )

  ELSE(Silo_LIBRARY)
    IF(Silo_FIND_REQURIED)
      message(SEND_ERROR "Unable to find the requested Silo libraries.")
    ENDIF(Silo_FIND_REQURIED)
  ENDIF(Silo_LIBRARY)
ENDIF(Silo_INCLUDE_DIR)

# handle the QUIETLY and REQUIRED arguments and set Silo_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(Silo DEFAULT_MSG Silo_LIBRARY Silo_INCLUDE_DIR)

MARK_AS_ADVANCED(
  Silo_INCLUDE_DIR
  Silo_LIBRARY
)
