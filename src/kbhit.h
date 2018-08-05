// simulate _kbhit() on Linux
// taken from http://linux-sxs.org/programming/kbhit.html

#ifndef KBHITh
#define KBHITh

#ifndef _MSC_VER
#ifndef __MINGW32__
//#ifndef __CYGWIN__

#include <termios.h>

class keyboard
{
  public:

     keyboard();
    ~keyboard();
    int kbhit();
    int getch();

  private:

    struct termios initial_settings, new_settings;
    int peek_character;

};

//#endif
#endif
#endif
#endif

