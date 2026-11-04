"""Compare centered 256/512 raw-image crops with identical local write masks."""
from .refine_conflicts import main


if __name__=='__main__':
    main(default_context_sizes=[256,512])
