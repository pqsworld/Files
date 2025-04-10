#ifdef MAKE_SO

#ifdef __cplusplus
extern "C" {
#endif

//#include<utils/Log.h>
//#include "alog.h"
#include <stdarg.h>
#include "android.h"
#include<utils/Log.h>
#undef LOG_TAG
#define LOG_TAG "silead_algo"

#undef LOG_NDEBUG
#define LOG_NDEBUG 0

#include "stdlib.h"


void* __real_malloc( size_t size );
void __real_free( void* ptr );
void* __real_realloc( void* ptr, size_t size );
void* __real_calloc( int nmemb, size_t size );

void* __attribute__( ( weak ) ) __wrap_malloc( size_t size )
{
    return __real_malloc( size );
}

void __attribute__( ( weak ) ) __wrap_free( void* ptr )
{
    __real_free( ptr );
}

void* __attribute__( ( weak ) ) __wrap_realloc( void* ptr, size_t size )
{
    return __real_realloc( ptr, size );
}

void* __attribute__( ( weak ) ) __wrap_calloc( int nmemb, size_t size )
{
    return __real_calloc( 1, nmemb * size );
}

void __attribute__( ( weak ) ) sil_loge( const char* fmt, ... )
{
    va_list argp;
    char log[1024] = { 0 };

    va_start( argp, fmt );
    vsnprintf( log, sizeof( log ) - 1, fmt, argp );
    va_end( argp );

    ALOGE( "[fp-algo] %s", log );
}

void __attribute__( ( weak ) ) sil_logd( const char* fmt, ... )
{
    va_list argp;
    char log[1024] = { 0 };

    va_start( argp, fmt );
    vsnprintf( log, sizeof( log ) - 1, fmt, argp );
    va_end( argp );

    ALOGE( "[fp-algo] %s", log );
}

void __attribute__( ( weak ) ) sil_logi( const char* fmt, ... )
{
    va_list argp;
    char log[1024] = { 0 };

    va_start( argp, fmt );
    vsnprintf( log, sizeof( log ) - 1, fmt, argp );
    va_end( argp );

    ALOGE( "[fp-algo] %s", log );
}

#ifdef __cplusplus
}
#endif
#endif /* MAKE_SO */
