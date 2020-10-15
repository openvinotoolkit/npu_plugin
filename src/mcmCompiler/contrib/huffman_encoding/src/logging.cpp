///
/// @file
/// @copyright All code copyright Movidius Ltd 2017, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief     logging functions.
///
///

#include <string>
#include <sstream> // ostringstream
#include <iostream>
#include <cstdio>
#include <cstdarg>
#include <cstdlib>
#include <ctime>

#include "logging.hpp" // Exception, VASPRINTF

int ARGS_verbosity = -1;

using std::ostringstream;
using std::string;

std::string LOGFILE = "c_model.log";

static void syslog(string const &);

/**
 * A utility routine which converts a printf-style call into the corresponding
 * string, and returns it.
 *
 * @param fmt  printf format
 * @return     The string equivalent
 */
string PrintToString(const char *fmt, ...)
{
   char *fmt_out = 0;
   va_list ap;
   va_start(ap, fmt);
   VASPRINTF(&fmt_out, fmt, ap);
   va_end(ap);

   if (!fmt_out)
      return ("Error: PrintToString");

   string output(fmt_out);
   free(fmt_out);
   return output;
} // PrintToString()

void Report(int level, std::stringstream &reportStream)
{
   if (level > ARGS_verbosity)
   {
      reportStream.str("");
      return; // exclude unimportant messages
   }

   std::cout << reportStream.str();
   reportStream.str("");
}

void Error(int level, std::stringstream &reportStream)
{
   // level is ignored here
   std::cerr << reportStream.str();
   reportStream.str("");
}

/**
 * Debug/log output routine. The first parameter is the 'importance' of the
 * message. Level 0 messages are the most important, and are always recorded
 * in the log file. Increasing levels correspond to increasingly lower
 * importances.
 *
 * The 'fmt' string, and any following arguments, must form a valid set of
 * arguments to 'printf'. If they don't, the call to 'vasprintf' is likely to
 * crash.
 *
 * @param level  Importance of this message; 0 is the most important
 * @param fmt    printf-style format for the message
 */
void Log(int level, const char *fmt, ...)
{
   if (level > ARGS_verbosity)
      return; // exclude unimportant messages

   // the current time
   char timebuf[128];
   time_t curtime = time(0);
   struct tm *loctime = localtime(&curtime);
   strftime(timebuf, 128, "%F %T", loctime);

   // get the formatted output
   char *fmt_out;
   va_list ap;
   va_start(ap, fmt);
   VASPRINTF(&fmt_out, fmt, ap);
   va_end(ap);

   ostringstream outstr;
   outstr << "(LOG: " << timebuf << ": Log L" << level << ")     "
          << fmt_out << '\n';
   syslog(outstr.str());
   if (fmt_out)
      free(fmt_out);
} // Log()

/**
 *
 */
void AssertFail(const char *file, int line, const char *fmt, ...)
{
   // get the formatted output first
   char *fmt_out;
   va_list ap;
   va_start(ap, fmt);
   VASPRINTF(&fmt_out, fmt, ap);
   va_end(ap);

   ostringstream outstr;
   outstr << "assertion failure at '" << file
          << "', line " << line << " (" << fmt_out << ")\n";
   // logging is handled at the catch
   // syslog(outstr.str());
   if (fmt_out)
      free(fmt_out);

   throw Exception(outstr.str());
} // AssertFail()

/**
 * Output a formatted message to the logfile.
 */
static void syslog(string const &str)
{
   FILE *ofile = fopen(LOGFILE.c_str(), "a");
   fprintf(ofile, "%s", str.c_str());
   fclose(ofile);
} // syslog()

/* ---------------------------------- EOF ---------------------------------- */
