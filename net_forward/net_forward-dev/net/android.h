#ifndef SL__HNDROID_H
#define SL__HNDROID_H
#include <stddef.h>

void __attribute__( ( weak ) ) sil_loge( const char* fmt, ... );

void __attribute__( ( weak ) ) sil_logd( const char* fmt, ... );

void __attribute__( ( weak ) ) sil_logi( const char* fmt, ... );

void* __attribute__( ( weak ) ) __wrap_malloc( size_t size );

void __attribute__( ( weak ) ) __wrap_free( void* ptr );
#endif
