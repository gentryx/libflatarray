// if we don't have a std library config set, try and find one:
#if !defined(BOOST_STDLIB_CONFIG) && !defined(BOOST_NO_STDLIB_CONFIG) && !defined(BOOST_NO_CONFIG) && defined(__cplusplus)
#  include <boost/config/select_stdlib_config.hpp>
#endif
// if we have a std library config, include it now:
#ifdef BOOST_STDLIB_CONFIG
#  include BOOST_STDLIB_CONFIG
#endif

// get config suffix code:
#include <boost/config/suffix.hpp>

