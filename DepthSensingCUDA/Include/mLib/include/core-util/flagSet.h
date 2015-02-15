#ifndef _UTIL_FLAGSET_H_
#define _UTIL_FLAGSET_H_


namespace ml {

	//! Strongly typed flag set used as bit mask over enums for function options
	//! based on discussion in http://stackoverflow.com/questions/4226960/type-safer-bitflags-in-c
	template <typename enumT>
	class FlagSet {
	public:
		typedef enumT enum_type;
		typedef typename std::underlying_type<enumT>::type store_type;

		// Default constructor (all 0s)
		FlagSet() : FlagSet(store_type(0)) { }

		// Value constructors
		FlagSet(store_type value) : flags_(value) { }
		FlagSet(enum_type value) : flags_(static_cast<store_type>(value)) { }

		// Explicit conversion operator
		operator store_type() const { return flags_; }

		operator std::string() const { return to_string(); }

		bool operator[] (enum_type flag) const { return test(flag); }

		FlagSet& operator= (store_type flag) {
			reset();
			set(flag);
			return *this;
		}

		std::string to_string() const {
			std::string str(size(), '0');
			for (size_t x = 0; x < size(); ++x) { str[size() - x - 1] = (flags_ & (1 << x) ? '1' : '0'); }
			return str;
		}

		FlagSet& set() {
			flags_ = ~store_type(0);
			return *this;
		}

		FlagSet& set(enum_type flag, bool val = true) {
			flags_ = (val ? (flags_ | flag) : (flags_ & ~flag));
			return *this;
		}

		FlagSet& reset() {
			flags_ = store_type(0);
			return *this;
		}

		FlagSet& reset(enum_type flag) {
			flags_ &= ~flag;
			return *this;
		}

		FlagSet& flip() {
			flags_ = ~flags_;
			return *this;
		}

		FlagSet& flip(enum_type flag) {
			flags_ ^= flag;
			return *this;
		}

		size_t count() const {
			// see http://www-graphics.stanford.edu/~seander/bithacks.html#CountBitsSetKernighan
			store_type bits = flags_;
			size_t total = 0;
			for (; bits != 0; ++total) {
				bits &= bits - 1; // clear the least significant bit set
			}
			return total;
		}

		/*constexpr*/ size_t size() const { // constexpr not supported in vs2010 yet
			return sizeof(enum_type) * 8;
		}

		bool test(enum_type flag) const { return (flags_ & flag) > 0; }

		bool any() const { return flags_ > 0; }

		bool none() const { return flags == 0; }

	private:
		store_type flags_;
	};

	template<typename enumT>
	FlagSet<enumT> operator& (const FlagSet<enumT>& lhs, const FlagSet<enumT>& rhs) {
		return FlagSet<enumT>(FlagSet<enumT>::store_type(lhs) & FlagSet<enumT>::store_type(rhs));
	}

	template<typename enumT>
	FlagSet<enumT> operator| (const FlagSet<enumT>& lhs, const FlagSet<enumT>& rhs) {
		return FlagSet<enumT>(FlagSet<enumT>::store_type(lhs) | FlagSet<enumT>::store_type(rhs));
	}

	template<typename enumT>
	FlagSet<enumT> operator^ (const FlagSet<enumT>& lhs, const FlagSet<enumT>& rhs) {
		return FlagSet<enumT>(FlagSet<enumT>::store_type(lhs) ^ FlagSet<enumT>::store_type(rhs));
	}

	template <class charT, class traits, typename enumT>
	std::basic_ostream<charT, traits>& operator<< (std::basic_ostream<charT, traits>& os, const FlagSet<enumT>& flagSet) {
		return os << flagSet.to_string();
	}

}  // namespace ml

#endif  // _UTIL_FLAGSET_H_
