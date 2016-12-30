#pragma once
#define CHECK_ERRORS()         \
  do {                         \
    GLenum err = glGetError(); \
    if (err) {                                                       \
      printf( "GL Error %x at line %d\n", (int)err, __LINE__);       \
      exit(-1);                                                      \
	    }                                                                \
    } while(0)