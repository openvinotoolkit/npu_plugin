///
/// @file
/// @copyright All code copyright Movidius Ltd 2017, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief     logging header file.
///
///

#ifndef LOGGING_HPP
#define LOGGING_HPP

#include <string>
#include <stdint.h>                     // uintnn_t


//#define LOGFILE "c_model.log"           //!< Output log file

#define MAX_VERBOSITY_LEVEL 5

using std::string;

#if defined(USE_INT128_CLASS) || defined(WIN32)
#include <cstdarg>
#define VASPRINTF ms_vasprintf            // 'vasprintf' is gnu-specific
extern int ms_vasprintf(char **ptr, const char *format, va_list ap);
#else
#define VASPRINTF vasprintf
#endif // WIN32/etc

// logging.cpp:
void Report( int level, std::stringstream &reportStream );
void Error ( int level, std::stringstream &reportStream );

extern std::string LOGFILE;

extern string PrintToString(const char *fmt, ...)
#ifndef WIN32
    __attribute__((format (printf, 1, 2)))
#endif
;
    
extern "C" {
	extern void AssertFail(const char*, int, const char *fmt, ...)
#ifndef WIN32
		__attribute__((format(printf, 3, 4), __noreturn__))
#endif
;
}

// globals
extern "C" {
   extern int ARGS_verbosity;                 //!< debug level; 0 off
}

/**
 * Logging routine. This is called with a log level, and a printf-style format
 * string and trailing parameters; the output is appended to file
 * 'c_model.log'. This routine is used within the library code for reporting
` * status and error information, and can also be called by library users to
 * add to the log file.
 *
 * 'level' should be an integer in the range [0,5]. Level 0 messages are
 * always added to the log file. Otherwise, the message is added only if
 * 'level' is less than or equal to the value set by 'setLogLevel'.  This
 * should be set to 0 to turn off all messaging apart from critical failure or
 * logging messages.
 */
extern "C" {
   extern void Log(int level, const char *fmt, ...)
#ifndef WIN32
      __attribute__((format (printf, 2, 3)))
#endif
;
}

/**
 * An exception class, derived from std::exception. All model exceptions are
 * reported using this class.
 */
class Exception: public std::exception {
public:
    /**
     * Ctor #1: C strings. This is required; ctor #2 won't work for C-strings
     *
     * @param message C-style string error message
     */
   explicit Exception(const char* message) :
         msg_(message)
       {
          Log(0, "Throwing exception, with message '%s'", message);
       }

    /**
     * Ctor #2: STL strings
     *
     * @param message Error message
     */
   explicit Exception(string const& message) :
         msg_(message)
       {
          Log(0, "Throwing exception, with message '%s'", msg_.c_str());
       }

   //! Destructor; virtual to allow for subclassing
   virtual ~Exception() throw() {}

    /**
     * Return a pointer to the error description
     *
     * @return The error message
     */
   virtual const char* what() const throw (){
      return msg_.c_str();
   }

protected:
   string msg_;
}; // class Exception

/**
 * Set the logging/debug level. This is normally set via the 'LogLevel' key in
 * the config file, but can also be set explicitly. 'level' should be set to 0
 * to turn off all messaging apart from critical failure or logging messages,
 * and can be given higher values to give progressively more log output. The
 * log output appears in file 'c_model.log'.
 *
 * @param level The required level
 */
extern void setLogLevel(int level);

#endif // LOGGING_HPP 

/* ---------------------------------- EOF ---------------------------------- */
