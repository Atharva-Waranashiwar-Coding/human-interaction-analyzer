export type ObjectReference = { id: string; class_name: string };
export type Annotation = {
  id?: string; start_time: number; end_time: number; behavior: string;
  participants: string[]; objects?: ObjectReference[]; parent_activity_id?: string | null;
  description: string; source: string; confidence?: number | null; review_status: string;
  annotation_kind?: 'behavior' | 'spatial' | 'activity' | 'audio';
  details?: Record<string, any>; original_suggestion?: Record<string, any>;
  review_history?: Record<string, any>[]; reviewer_feedback?: string;
  rule_version?: string | null; evaluation_role?: 'unspecified' | 'ground_truth';
};
export type Region = { id: string; label: string; kind: string; box: number[] };
