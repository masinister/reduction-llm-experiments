"""Merge strategies for combining partial structured outputs.

This module provides different strategies for merging chunk-level outputs:
  - Incremental merging: Merge outputs as they arrive, maintaining constant memory
  - Hierarchical merging: Tree-reduce pattern for batch processing
  
Both strategies use a pluggable merge_fn callback for the actual LLM merge operation.
"""

from typing import List, Callable, Optional
import logging

logger = logging.getLogger(__name__)


class IncrementalMerger:
    """Incremental merge strategy that compacts outputs as they arrive.
    
    This strategy maintains a sliding window of unmerged items. When the buffer
    reaches batch_size, it merges the oldest items and replaces them with a 
    single merged output. This keeps memory usage bounded.
    """

    def __init__(
        self,
        merge_fn: Callable[[List[str], str], str],
        batch_size: int = 5
    ):
        """Initialize incremental merger.
        
        Args:
            merge_fn: Callback function with signature:
                      (batch_outputs: List[str], running_summary: str) -> merged_text: str
                      This function should call the LLM to merge the batch.
            batch_size: Number of items to accumulate before triggering a merge
        """
        self.merge_fn = merge_fn
        self.batch_size = batch_size
        self.items: List[str] = []

    def add_and_maybe_compact(self, item: str, running_summary: str = "") -> None:
        """Add a new item and potentially trigger compaction.
        
        Args:
            item: New chunk output to add
            running_summary: Current summary for context during merging
        """
        if not item or not item.strip():
            logger.warning("add_and_maybe_compact: skipping empty item")
            return
        
        self.items.append(item)
        
        # Trigger compaction when buffer is full
        if len(self.items) >= self.batch_size:
            # Take the first batch_size items, merge them
            batch = self.items[:self.batch_size]
            
            try:
                merged = self.merge_fn(batch, running_summary)
                
                # Replace batch with merged result
                self.items = [merged] + self.items[self.batch_size:]
                
                logger.debug(
                    "Compacted %d items into 1 merged output (%d items remaining)",
                    len(batch), len(self.items)
                )
            except Exception as e:
                logger.exception("Compaction failed: %s", e)
                # Keep original items if merge fails

    def finalize(self, running_summary: str = "") -> str:
        """Merge all remaining items into a final output.
        
        Args:
            running_summary: Current summary for context during final merge
            
        Returns:
            Final merged text
        """
        if not self.items:
            logger.warning("finalize: no items to merge")
            return ""
        
        if len(self.items) == 1:
            return self.items[0]
        
        # Hierarchically merge remaining items
        return self._hierarchical_merge(self.items, running_summary)

    def _hierarchical_merge(
        self,
        items: List[str],
        running_summary: str
    ) -> str:
        """Recursively merge items using tree-reduce pattern.
        
        Args:
            items: List of items to merge
            running_summary: Current summary for context
            
        Returns:
            Single merged output
        """
        if len(items) == 0:
            return ""
        if len(items) == 1:
            return items[0]
        
        # Batch and recurse
        next_level: List[str] = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            try:
                merged = self.merge_fn(batch, running_summary)
                next_level.append(merged)
            except Exception as e:
                logger.exception("Hierarchical merge failed for batch starting at %d: %s", i, e)
                # Fallback: concatenate batch as JSON array
                next_level.append("[" + ",".join(batch) + "]")
        
        # Recurse until single item remains
        return self._hierarchical_merge(next_level, running_summary)


class HierarchicalMerger:
    """Pure hierarchical (tree-reduce) merge strategy.
    
    This strategy waits until all chunk outputs are available, then merges them
    in a tree-reduce pattern. More deterministic than incremental, but requires
    holding all outputs in memory.
    """

    def __init__(
        self,
        merge_fn: Callable[[List[str], str], str],
        batch_size: int = 5
    ):
        """Initialize hierarchical merger.
        
        Args:
            merge_fn: Callback function with signature:
                      (batch_outputs: List[str], running_summary: str) -> merged_text: str
            batch_size: Maximum items to merge in a single operation
        """
        self.merge_fn = merge_fn
        self.batch_size = batch_size

    def merge_all(
        self,
        items: List[str],
        running_summary: str = ""
    ) -> str:
        """Merge all items using tree-reduce pattern.
        
        Args:
            items: All chunk outputs to merge
            running_summary: Optional summary for context
            
        Returns:
            Final merged output
        """
        # Filter out empty items
        valid_items = [item for item in items if item and item.strip()]
        
        if not valid_items:
            logger.warning("merge_all: no valid items to merge")
            return ""
        
        if len(valid_items) == 1:
            return valid_items[0]
        
        return self._hierarchical_merge(valid_items, running_summary)

    def _hierarchical_merge(
        self,
        items: List[str],
        running_summary: str
    ) -> str:
        """Recursively merge items in batches.
        
        Args:
            items: List of items to merge
            running_summary: Current summary for context
            
        Returns:
            Single merged output
        """
        if len(items) == 0:
            return ""
        if len(items) == 1:
            return items[0]
        
        # Single batch - merge directly
        if len(items) <= self.batch_size:
            try:
                return self.merge_fn(items, running_summary)
            except Exception as e:
                logger.exception("Direct merge failed: %s", e)
                # Fallback: return first item (conservative)
                return items[0]
        
        # Multiple batches - tree reduce
        next_level: List[str] = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            try:
                merged = self.merge_fn(batch, running_summary)
                next_level.append(merged)
            except Exception as e:
                logger.exception("Batch merge failed for batch starting at %d: %s", i, e)
                # Fallback: include first item from failed batch
                next_level.append(batch[0] if batch else "")
        
        # Recurse
        return self._hierarchical_merge(next_level, running_summary)
